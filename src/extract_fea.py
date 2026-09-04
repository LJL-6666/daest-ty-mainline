 
import numpy as np
from data.io_utils import (
    load_finetune_EEG_data,
    get_load_data_func,
    load_processed_SEEDV_NEW_data,
    get_ty6_save_base,
    _ty_save_name,
    pop_last_ty_allsubs_labels,
)
from utils.utils import pl_accelerator_devices
from data.data_process import running_norm_onesubsession, LDS, LDS_acc
from utils.reorder_vids import (
    video_order_load,
    reorder_vids_sepVideo,
    reorder_vids_back,
    ty_subject_ids_from_pkl,
)
import hydra
from omegaconf import DictConfig
from model import ExtractorModel
from model.channel_adapt import (
    adapt_extractor_inplace,
    get_preset_select_idx,
    reorder_data_ty_pkl_to_faced_by_name,
)
from data.dataset import SEEDV_Dataset 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import os
from tqdm import tqdm
import logging
import mne
import glob
import pickle as pkl
import gc

log = logging.getLogger(__name__)


def _ty_questionnaire_reorder_features(cfg, load_dir, fea, n_vids):
    """TY 问卷播放序重排 + 后续 reorder_vids_back（与 FACED After_remarks 流程一致）。"""
    questionnaire_dir = getattr(cfg.data, 'questionnaire_dir', None) or os.environ.get('CLISA_QUESTIONNAIRE_DIR', '')
    if not questionnaire_dir or not os.path.isdir(os.path.expanduser(questionnaire_dir)):
        log.info('TY6 n_vids=%s: no questionnaire_dir set or dir missing, skip reorder', n_vids)
        return fea, None
    pkl_path = os.path.join(load_dir, getattr(cfg.data, 'pkl_file', 'data_6class_communication.pkl'))
    with open(pkl_path, 'rb') as f:
        raw = pkl.load(f)
    subject_ids = ty_subject_ids_from_pkl(raw, cfg.data.n_subs)
    task = getattr(cfg.data, 'questionnaire_task', 'exp1')
    vid_order = video_order_load(
        n_vids, dataset='TY6', questionnaire_dir=questionnaire_dir,
        subject_ids=subject_ids, questionnaire_task=task,
    )
    vid_inds = np.arange(n_vids)
    fea, vid_play_order_new = reorder_vids_sepVideo(fea, vid_order, vid_inds, n_vids)
    log.info('TY6 reorder from questionnaire CSV (%s videos, task=%s)', n_vids, task)
    return fea, vid_play_order_new


def _predict_features(Extractor, fold_loader, cfg, fold):
    """Run feature extraction without Lightning's prediction accumulator.

    This keeps preprocessing / smoothing on CPU and only uses GPU for the
    forward pass, which avoids the large CPU-side accumulation that previously
    caused OOM in later folds.
    """
    use_cpu = bool(getattr(cfg.train, 'use_cpu', False))
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    Extractor = Extractor.to(device)
    Extractor.eval()
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(fold_loader, desc=f'Predict fold {fold}', leave=False):
            data, _ = batch
            data = data.to(device, non_blocking=not use_cpu)
            fea_batch = Extractor(data)
            outputs.append(fea_batch.detach().cpu().numpy())
            del data, fea_batch

    pred = np.concatenate(outputs, axis=0)
    del outputs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return pred

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    use_running_norm = bool(getattr(cfg.ext_fea, 'use_running_norm', True))
    use_lds = bool(getattr(cfg.ext_fea, 'use_lds', True))
    lds_v0 = float(getattr(cfg.ext_fea, 'lds_v0', 0.01))
    lds_t = float(getattr(cfg.ext_fea, 'lds_t', 0.0001))
    lds_sigma = float(getattr(cfg.ext_fea, 'lds_sigma', 1.0))
    lds_given_all = int(getattr(cfg.ext_fea, 'lds_given_all', 1))
    ckpt_run = int(getattr(cfg.ext_fea, 'ckpt_run', None) or cfg.log.run)
    log.info(f'Smoothing switches: running_norm={use_running_norm}, lds={use_lds}')
    if use_lds:
        log.info(f'LDS params: V0={lds_v0}, T={lds_t}, sigma={lds_sigma}, given_all={lds_given_all}')
    # TY6: data_dir 为含 pkl 的目录，无 processed_data；save 可用 work_dir 统一到 work_dir/dataset_name/ext_fea
    load_dir = cfg.data.data_dir if getattr(cfg.data, 'dataset_name', None) == 'TY6' else os.path.join(cfg.data.data_dir, 'processed_data')
    data2, onesub_label2, n_samples2_onesub, n_samples2_sessions = load_finetune_EEG_data(load_dir, cfg.data)
    allsubs_label2 = pop_last_ty_allsubs_labels()
    #data2 shape (n_subs,session*vid*n_samples, n_chans, n_pionts)
    data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-2], data2.shape[-1])

    # 跨库迁移：按电极名通道适配（ty_pkl_to_faced_by_name）
    ch_adapt_cfg = getattr(cfg.ext_fea, 'channel_adapt', None)
    ch_adapt_enabled = bool(ch_adapt_cfg is not None and getattr(ch_adapt_cfg, 'enable', False))
    ch_adapt_weight_idx = None
    ch_adapt_target_n = None
    if ch_adapt_enabled:
        ch_adapt_mode = str(getattr(ch_adapt_cfg, 'mode', 'ty_pkl_to_faced_by_name'))
        _data_idx, ch_adapt_weight_idx, ch_adapt_target_n = get_preset_select_idx(ch_adapt_mode)
        log.info(
            f"channel_adapt enabled: mode={ch_adapt_mode}, "
            f"weight_idx_len={len(ch_adapt_weight_idx)}, target_n_channs={ch_adapt_target_n}"
        )
        old_shape = data2.shape
        data2 = reorder_data_ty_pkl_to_faced_by_name(data2, channel_axis=-2)
        log.info(f"channel_adapt data reorder: {old_shape} -> {data2.shape}")

    save_dir = os.path.join(get_ty6_save_base(cfg.data, getattr(cfg.log, 'cp_dir', None)), 'ext_fea', f'fea_r{cfg.log.run}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    np.save(save_dir+'/onesub_label2.npy',onesub_label2)
    if allsubs_label2 is not None:
        if allsubs_label2.shape != (cfg.data.n_subs, data2.shape[1]):
            raise ValueError(
                f'allsubs_label2 shape={allsubs_label2.shape}，期望 '
                f'({cfg.data.n_subs}, {data2.shape[1]})'
            )
        allsubs_path = save_dir + '/allsubs_label2.npy'
        tmp_path = allsubs_path + '.tmp.npy'
        np.save(tmp_path, allsubs_label2)
        os.replace(tmp_path, allsubs_path)
        log.info('saved per-subject finetune labels: %s', allsubs_path)
        # 特征抽取 DataLoader 标签仅占位；用真实逐被试标签避免误导排查
        onesub_label2_for_loader = allsubs_label2.reshape(-1)
    else:
        onesub_label2_for_loader = None

    # 外源 ckpt（FACED→TY）：load_cp_dir / ckpt_dataset / ckpt_fold 可覆盖默认路径
    ckpt_cp_dir = getattr(cfg.ext_fea, 'load_cp_dir', None) or getattr(cfg.log, 'cp_dir', None)
    ckpt_dataset = getattr(cfg.ext_fea, 'ckpt_dataset', None) or _ty_save_name(cfg.data)
    ckpt_fold_override = getattr(cfg.ext_fea, 'ckpt_fold', None)

    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs

    n_per = round(cfg.data.n_subs / n_folds)
    fold_list = list(range(0, n_folds))
    if getattr(cfg.train, 'fold_index', None) is not None:
        fold_list = [cfg.train.fold_index]
        assert 0 <= fold_list[0] < n_folds, f"train.fold_index={fold_list[0]} 应在 [0, n_folds) 内"
    
    for fold in fold_list:
        save_path = os.path.join(save_dir,cfg.log.exp_name+'_r'+str(cfg.log.run)+f'_f{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        if os.path.exists(save_path):
            log.info(f'skip fold {fold}: feature file already exists at {save_path}')
            continue
        log.info(f"fold:{fold}")
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        # if len(val_subs) == 1:
        #     val_subs = list(val_subs) + train_subs
        log.info(f'train_subs:{train_subs}')
        log.info(f'val_subs:{val_subs}' )
        


        data2_train = data2[train_subs] # (subs,vid*n_samples, 62, 1250)
        
        # print(data2[0,0])
        if cfg.ext_fea.normTrain:
            data2_fold = normTrain(data2,data2_train)
        else:
            log.info('no normTrain')
            data2_fold = data2
        # print(data2_fold[0,0])
        if cfg.ext_fea.use_pretrain:
            log.info('Use pretrain model:')
            data2_fold = data2_fold.reshape(-1, data2_fold.shape[-2], data2_fold.shape[-1])
            if onesub_label2_for_loader is not None:
                label2_fold = onesub_label2_for_loader
            else:
                label2_fold = np.tile(onesub_label2, cfg.data.n_subs)
            foldset = SEEDV_Dataset(data2_fold, label2_fold)
            del data2_fold, label2_fold
            fold_loader = DataLoader(foldset, batch_size=cfg.ext_fea.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
            source_fold = fold if ckpt_fold_override is None else int(ckpt_fold_override)
            checkpoint_pattern = os.path.join(ckpt_cp_dir, ckpt_dataset, f'r{ckpt_run}', f'f{source_fold}epoch=*.ckpt')
            checkpoint_list = glob.glob(checkpoint_pattern)
            if not checkpoint_list:
                checkpoint_pattern = os.path.join(ckpt_cp_dir, ckpt_dataset, f'r{ckpt_run}', f'f{source_fold}*')
                checkpoint_list = glob.glob(checkpoint_pattern)
            if not checkpoint_list:
                # 相对路径下找不到时，在 runs/<dataset_name>/ 下自动查找该数据集最近的 run 中的 checkpoint
                try:
                    # 当前 cwd 为 runs/DATASET/date/time_run1，上两级得到 runs/DATASET
                    base_runs = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
                    save_name = _ty_save_name(cfg.data)
                    if os.path.basename(base_runs) in (cfg.data.dataset_name, save_name):
                        fallback_pattern = os.path.join(base_runs, '*', '*', 'runs', 'cp', save_name, f'r{ckpt_run}', f'f{source_fold}*')
                        checkpoint_list = glob.glob(fallback_pattern)
                        if checkpoint_list:
                            checkpoint_list = [(os.path.getmtime(p), p) for p in checkpoint_list]
                            checkpoint_list.sort(key=lambda x: -x[0])
                            checkpoint_list = [p for _, p in checkpoint_list]
                except Exception:
                    pass
            if not checkpoint_list:
                raise FileNotFoundError(
                    f'未找到 fold {fold} 的 checkpoint（source_fold={source_fold}），请确认 train_ext 已跑完且 log.cp_dir/ext_fea.load_cp_dir 指向正确。\n'
                    f'  查找路径: {checkpoint_pattern}\n'
                    f'  若 train_ext 与 extract_fea 分开运行，请用绝对路径指定 log.cp_dir，例如：\n'
                    f'  log.cp_dir=/path/to/FACED-base/runs/TY9_movie_ME_cp/runs/cp'
                )
            checkpoint = checkpoint_list[0]
            if len(checkpoint_list) > 1:
                # 同折多 ckpt（如 f2epoch=0 与 f2epoch=2）时取 epoch 最大者
                def _epoch_key(p):
                    import re
                    m = re.search(r'epoch=(\d+)', os.path.basename(p))
                    return int(m.group(1)) if m else -1
                checkpoint = max(checkpoint_list, key=_epoch_key)
            
            log.info('checkpoint load from: '+checkpoint)
            # 多 GPU 保存的 ckpt 在单卡/单进程下需 map_location，否则会报 "deserialize on CUDA device X but device_count() is 1"
            Extractor = ExtractorModel.load_from_checkpoint(checkpoint_path=checkpoint, map_location="cpu")
            Extractor.model.stratified = []
            if ch_adapt_enabled and ch_adapt_weight_idx is not None and ch_adapt_target_n is not None:
                _diffs = adapt_extractor_inplace(
                    Extractor,
                    weight_select_idx=list(ch_adapt_weight_idx),
                    target_n_channs=int(ch_adapt_target_n),
                )
                log.info(f"channel_adapt weight slicing applied, layers changed: {len(_diffs)}")
            log.info('load model:'+checkpoint)
            pred = _predict_features(Extractor, fold_loader, cfg, fold)
            log.debug(pred.shape)
            # pred = pred

            # max_fea = np.max(pred)
            # min_fea = np.min(pred)
            # print(max_fea,min_fea)
            # if np.isinf(pred).any():
            #     print("There are inf values in the array")

            fea = cal_fea(pred,cfg.ext_fea.mode)
            # print('fea0:',fea[0])
            fea = fea.reshape(cfg.data.n_subs,-1,fea.shape[-1])
            
        else:
            #data2_fold shape (n_subs,session*vid*n_samples, n_chans, n_pionts)
            log.info('Direct DE extraction:')
            n_subs, n_samples, n_chans, sfreqs = data2_fold.shape
            freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]
            de_data = np.zeros((n_subs, n_samples, n_chans, len(freqs)))
            n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
            
            for idx, band in enumerate(freqs):
                for sub in range(n_subs):
                    log.debug(f'sub:{sub}')
                    for vid in tqdm(range(len(n_samples2_onesub)), desc=f'Direct DE Processing sub: {sub}', leave=False):
                        data_onevid = data2_fold[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]]
                        data_onevid = data_onevid.transpose(1,0,2)
                        data_onevid = data_onevid.reshape(data_onevid.shape[0],-1)
                        
                        data_video_filt = mne.filter.filter_data(data_onevid, sfreqs, l_freq=band[0], h_freq=band[1])
                        data_video_filt = data_video_filt.reshape(n_chans, -1, sfreqs)
                        de_onevid = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data_video_filt, 2))).T
                        de_data[sub,  n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1], :, idx] = de_onevid
            fea = de_data.reshape(n_subs, n_samples, -1)
        log.debug(fea.shape)    
        
        fea_train = fea[train_subs]
        
        data_mean = np.mean(np.mean(fea_train, axis=1),axis=0)
        data_var = np.mean(np.var(fea_train, axis=1),axis=0)
        # print('fea_mean:',data_mean) 
        # print('fea_var:',data_var)
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')
            
        # reorder（FACED 用 After_remarks；TY6 用问卷 CSV，与 Clisa_6-all 一致）
        vid_play_order_new = None
        if cfg.data.dataset_name == 'FACED':
            after_remarks = getattr(cfg.data, 'after_remarks_dir', None)
            vid_order = video_order_load(cfg.data.n_vids, after_remarks_dir=after_remarks)
            if cfg.data.n_class == 2:
                n_vids = 24
            elif cfg.data.n_class == 9:
                n_vids = 28
            vid_inds = np.arange(n_vids)
            fea, vid_play_order_new = reorder_vids_sepVideo(fea, vid_order, vid_inds, n_vids)
        elif cfg.data.dataset_name == 'TY6' and getattr(cfg.data, 'n_vids', None) == 19:
            fea, vid_play_order_new = _ty_questionnaire_reorder_features(cfg, load_dir, fea, 19)
        elif cfg.data.dataset_name == 'TY6' and getattr(cfg.data, 'n_vids', None) == 28:
            fea, vid_play_order_new = _ty_questionnaire_reorder_features(cfg, load_dir, fea, 28)


        n_sample_sum_sessions = np.sum(n_samples2_sessions,1)
        n_sample_sum_sessions_cum = np.concatenate((np.array([0]), np.cumsum(n_sample_sum_sessions)))

        # fea_processed = np.zeros_like(fea)
        # rn_scope: 'session'(默认，原行为，跨试次连续累积) | 'video'(每试次独立重置)
        rn_scope = str(getattr(cfg.ext_fea, 'rn_scope', 'session')).lower()
        if use_running_norm and rn_scope == 'video':
            log.info('running norm (scope=video, 每试次独立重置):')
            _vid_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
            n_samples2_onesub_cum = _vid_cum
            for sub in range(cfg.data.n_subs):
                log.debug(f'sub:{sub}')
                for vid in tqdm(range(len(n_samples2_onesub)), desc=f'running norm(video) sub: {sub}', leave=False):
                    a, b = n_samples2_onesub_cum[vid], n_samples2_onesub_cum[vid+1]
                    fea[sub, a:b] = running_norm_onesubsession(
                                                fea[sub, a:b], data_mean, data_var, cfg.ext_fea.rn_decay)
        elif use_running_norm:
            log.info('running norm (scope=session, 跨试次连续):')
            for sub in range(cfg.data.n_subs):
                log.debug(f'sub:{sub}')
                for s in  tqdm(range(len(n_sample_sum_sessions)), desc=f'running norm sub: {sub}', leave=False):
                    fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]] = running_norm_onesubsession(
                                                fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]],
                                                data_mean,data_var,cfg.ext_fea.rn_decay)
        else:
            log.info('skip running norm (ext_fea.use_running_norm=False)')
                
        # print('rn:',fea[0,0])
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')

        # order back
        if cfg.data.dataset_name == 'FACED' and vid_play_order_new is not None:
            fea = reorder_vids_back(fea, len(vid_inds), vid_play_order_new)
        elif cfg.data.dataset_name == 'TY6' and vid_play_order_new is not None:
            n_vids_back = int(getattr(cfg.data, 'n_vids', 19))
            fea = reorder_vids_back(fea, n_vids_back, vid_play_order_new)
        
        n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
        # LDS
        if use_lds:
            log.info('LDS:')
            for sub in range(cfg.data.n_subs):
                log.debug(f'sub:{sub}')
                for vid in tqdm(range(len(n_samples2_onesub)), desc=f'LDS Processing sub: {sub}', leave=False):
                    fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]] = LDS(
                        fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]],
                        V0=lds_v0,
                        T=lds_t,
                        sigma=lds_sigma,
                        given_all=lds_given_all,
                    )
        else:
            log.info('skip LDS smoothing (ext_fea.use_lds=False)')
        fea = fea.reshape(-1,fea.shape[-1])
        fea = fea.astype(np.float32, copy=False)
        
        
        # (8.32145433e-18-8.31764020e-18)/np.sqrt(4.01888196e-40)
        
        # max_fea = np.max(fea)
        # min_fea = np.min(fea)
        # print(max_fea,min_fea)
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')

        # if not os.path.exists(cfg.ext_fea.save_dir):
        #     os.makedirs(cfg.ext_fea.save_dir)  
        np.save(save_path,fea)
        log.info(f'fea saved to {save_path}')

        del fea, fea_train
        if cfg.ext_fea.use_pretrain:
            del pred, fold_loader, foldset, Extractor
        else:
            del de_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if cfg.train.iftest :
            log.info('test mode!')
            break

    
def normTrain(data2,data2_train):
    log.info('normTrain')
    temp = np.transpose(data2_train,(0,1,3,2))
    temp = temp.reshape(-1,temp.shape[-1])
    data2_mean = np.mean(temp, axis=0)
    data2_var = np.var(temp, axis=0)
    data2_normed = (data2 - data2_mean.reshape(-1,1)) / np.sqrt(data2_var + 1e-5).reshape(-1,1)
    return data2_normed

def cal_fea(data,mode):
    if mode == 'de':
        # print(np.var(data, 3).squeeze()[0])
        fea = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data, 3))).squeeze()
        # fea[fea<-40] = -40
    elif mode == 'me':
        fea = np.mean(data, axis=3).squeeze()
    # print(fea.shape)
    # print(fea[0])
    return fea




if __name__ == '__main__':
    ext_fea()
