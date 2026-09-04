import os
import numpy as np
import scipy.io as sio
import re
import pickle as pkl
import logging

# 播放序重排只发生在 extract_fea（与 FACED 一致），加载期不再引入 reorder 依赖

log_io = logging.getLogger(__name__)

# TY 6 分类 19 视频标签（与 Clisa_6-all/config_6class.LABEL_6 一致）
LABEL_6 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
# TY 观影 9 分类 28 视频标签（与 FACED 9 类 / 问卷 videoEmotion 顺序一致）
LABEL_9 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]



def robust_zscore_per_subject(x, thr_mult=30.0):
    """逐被试稳健 z-score（与 FACED 加载期归一化同一实现）。

    以 thr = thr_mult * median(|x|) 为幅值阈，仅用阈内样本估计 mean/std，
    从而不被个别被试的量纲错误 / 大幅伪迹带偏。x 为单被试全部数据。
    """
    x = np.asarray(x, dtype=np.float64)
    thr = thr_mult * np.median(np.abs(x))
    inlier = x[np.abs(x) < thr] if thr > 0 else x
    if inlier.size == 0:
        inlier = x
    mu = np.mean(inlier)
    sd = np.std(inlier)
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (x - mu) / sd


def _ty_save_name(cfg):
    """可选 save_dataset_name，用于隔离不同 TY 实验的写出目录；默认仍用 dataset_name。"""
    return getattr(cfg, 'save_dataset_name', None) or getattr(cfg, 'dataset_name', None)


def get_ty6_save_base(cfg, cp_dir=None):
    """
    TY6 数据写入根目录：
    - 若传入 cp_dir，则用 dirname(cp_dir)/save_name，保证 extract_fea 与 train_mlp 用同一路径（不依赖 Hydra run 目录）。
    - 否则若配置了 work_dir 则写入 work_dir/save_name，否则用 data_dir。
    cfg 为 data 配置（含 data_dir, dataset_name, work_dir, 可选 save_dataset_name）。
    """
    if getattr(cfg, 'dataset_name', None) != 'TY6':
        return cfg.data_dir
    save_name = _ty_save_name(cfg)
    if cp_dir:
        base = os.path.dirname(os.path.abspath(cp_dir))
        return os.path.join(base, save_name)
    work_dir = getattr(cfg, 'work_dir', None)
    if not work_dir:
        return cfg.data_dir
    # work_dir 为相对路径时相对于当前工作目录（建议在 base 目录下执行）
    base = os.path.abspath(os.path.expanduser(work_dir))
    if not os.path.isabs(work_dir):
        base = os.path.join(os.getcwd(), work_dir)
    return os.path.join(base, save_name)

def get_load_data_func(dataset_name):
    if dataset_name == 'SEEDV':
        return load_processed_SEEDV_NEW_data
    elif dataset_name == 'SEED':
        return load_processed_SEED_NEW_data
    elif dataset_name == 'FACED':
        return load_processed_FACED_NEW_data
    elif dataset_name == 'TY6':
        return load_processed_TY6_pkl_data
    else:
        raise ValueError('dataset_name not found')

def load_EEG_data(data_dir, cfg):
    if cfg.dataset_name == 'TY6':
        pkl_path = data_dir if (isinstance(data_dir, str) and data_dir.endswith('.pkl')) else os.path.join(data_dir, getattr(cfg, 'pkl_file', 'data_6class_communication.pkl'))
        data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_TY6_pkl_data(
            pkl_path, cfg.fs, cfg.n_channs, cfg.timeLen, cfg.timeStep,
            cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class,
            questionnaire_dir=getattr(cfg, 'questionnaire_dir', None),
            questionnaire_task=getattr(cfg, 'questionnaire_task', None),
            per_subject_norm=bool(getattr(cfg, 'per_subject_norm', True)),
        )
        return data, onesub_labels, n_samples_onesub, n_samples_sessions
    load_data_func = get_load_data_func(cfg.dataset_name)
    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_data_func(
                                data_dir, cfg.fs, cfg.n_channs, cfg.timeLen, cfg.timeStep, 
                                cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class)
    return data, onesub_labels, n_samples_onesub, n_samples_sessions

def load_finetune_EEG_data(data_dir, cfg):
    if cfg.dataset_name == 'TY6':
        pkl_path = data_dir if (isinstance(data_dir, str) and data_dir.endswith('.pkl')) else os.path.join(data_dir, getattr(cfg, 'pkl_file', 'data_6class_communication.pkl'))
        data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_TY6_pkl_data(
            pkl_path, cfg.fs, cfg.n_channs, cfg.timeLen2, cfg.timeStep2,
            cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class,
            questionnaire_dir=getattr(cfg, 'questionnaire_dir', None),
            questionnaire_task=getattr(cfg, 'questionnaire_task', None),
            per_subject_norm=bool(getattr(cfg, 'per_subject_norm', True)),
        )
        return data, onesub_labels, n_samples_onesub, n_samples_sessions
    load_data_func = get_load_data_func(cfg.dataset_name)
    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_data_func(
                                data_dir, cfg.fs, cfg.n_channs, cfg.timeLen2, cfg.timeStep2, 
                                cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class)
    return data, onesub_labels, n_samples_onesub, n_samples_sessions


def _resolve_ty_video_labels(raw, n_vids, n_class):
    """优先使用 pkl 内标签；否则按视频数回退到 LABEL_6 / LABEL_9。

    若存在逐被试标签 label_by_subject_video，则额外返回 (n_subs, n_vids) 矩阵；
    onesub 侧仍用 label_by_video / 多数票，供 ME 预训练 sliced cache 兼容。
    """
    subject_video = None
    if raw.get('label_by_subject_video') is not None:
        mat = np.asarray(raw['label_by_subject_video'])
        if mat.ndim != 2 or mat.shape[1] != n_vids:
            raise ValueError(
                f'TY pkl label_by_subject_video shape={mat.shape}，期望 (*, {n_vids})'
            )
        subject_video = mat.astype(np.int64, copy=False)

    for key in ('label_9_by_video', 'label_by_video', 'labels_by_video'):
        if key in raw and raw[key] is not None:
            labels = list(raw[key])
            if len(labels) != n_vids:
                raise ValueError(
                    f'TY pkl 字段 {key} 长度={len(labels)}，与 n_vids={n_vids} 不一致'
                )
            return labels, key, subject_video
    if subject_video is not None:
        # 多数票兜底，仅用于 ME 预训练 metadata（对比学习本身不用情绪类标签）
        # 忽略 -1（全 0 无效）；该视频全员无效时回退 0
        labels = []
        for j in range(n_vids):
            valid = subject_video[:, j]
            valid = valid[valid >= 0]
            if valid.size == 0:
                labels.append(0)
                continue
            vals, counts = np.unique(valid, return_counts=True)
            labels.append(int(vals[np.argmax(counts)]))
        return labels, 'majority_of_label_by_subject_video', subject_video
    if n_vids == 19 and n_class == 6:
        return list(LABEL_6), 'LABEL_6', None
    if n_vids == 28 and n_class == 9:
        return list(LABEL_9), 'LABEL_9', None
    raise ValueError(
        f'TY pkl 缺少视频标签，且无法从 n_vids={n_vids}/n_class={n_class} 推断；'
        '请在 pkl 中提供 label_9_by_video / label_by_video / label_by_subject_video。'
    )


def load_processed_TY6_pkl_data(pkl_path, fs, n_chans, timeLen, timeStep, n_session=1,
                                 n_subs=None, n_vids=19, n_class=6, t=30,
                                 questionnaire_dir=None, questionnaire_task=None,
                                 per_subject_norm=True):
    """
    从 TY 单文件 pkl 加载数据。
    支持:
      - 6 类 19 视频: {'data': (n_subs, 19, 31, 7500), ...}
      - 9 类 28 视频: {'data': (n_subs, 28, 31, 7500), 'label_9_by_video': list, ...}
    输出与 load_processed_FACED_NEW_data 一致: data (N, n_chans, points_len), onesub_labels, n_samples_onesub, n_samples_sessions
    """
    with open(pkl_path, 'rb') as f:
        raw = pkl.load(f)
    data_all = raw['data']  # (n_subs, n_vids, 31, 7500)
    if data_all.ndim != 4:
        raise ValueError(f'TY pkl data 维度错误: shape={data_all.shape}，期望 (n_subs, n_vids, n_chans, n_times)')
    n_in_pkl, n_vids_pkl = data_all.shape[0], data_all.shape[1]
    if n_vids is not None and int(n_vids) != int(n_vids_pkl):
        raise ValueError(
            f'TY pkl 视频数={n_vids_pkl}，与配置 n_vids={n_vids} 不一致。'
            '请检查是否误用 6 类/9 类输入。'
        )
    n_vids = int(n_vids_pkl)
    if n_subs is not None:
        if n_subs > n_in_pkl:
            raise ValueError(
                f'TY6 pkl 被试数不足: pkl 有 {n_in_pkl} 人，配置 n_subs={n_subs}。'
                '请将 n_subs 设为 pkl 实际被试数或更小。'
            )
        if n_subs < n_in_pkl:
            log_io.warning('TY6 使用被试子集: pkl 有 %s 人，配置 n_subs=%s，取前 n_subs 人。', n_in_pkl, n_subs)
            data_all = data_all[:n_subs]
    n_subs_actual = data_all.shape[0]
    n_points_per_vid = int(t * fs)  # 7500
    if data_all.shape[-1] != n_points_per_vid:
        raise ValueError(
            f'TY pkl 每视频点数={data_all.shape[-1]}，与 fs*t={n_points_per_vid} 不一致'
        )

    # 逐被试稳健归一化：与 FACED loader 同一实现，抵消个别被试量纲/伪迹造成的幅值差
    if per_subject_norm:
        data_all = np.asarray(data_all, dtype=np.float64)
        _sd_before = []
        for _i in range(data_all.shape[0]):
            _sd_before.append(float(np.std(data_all[_i])))
            data_all[_i] = robust_zscore_per_subject(data_all[_i])
        log_io.info(
            'TY per-subject robust z-score applied: raw std min=%.3g max=%.3g -> normalized',
            float(np.min(_sd_before)), float(np.max(_sd_before)),
        )
    else:
        log_io.info('TY per-subject robust z-score skipped (data.per_subject_norm=false)')

    video_labels, label_src, subject_video = _resolve_ty_video_labels(raw, n_vids, n_class)
    if subject_video is not None and subject_video.shape[0] > n_subs_actual:
        subject_video = subject_video[:n_subs_actual]

    # 说明: TY pkl 第一维已是 videoIndex 槽位（与 label_9_by_video / label_by_subject_video 对齐）。
    # 2026-08-19 曾在此按问卷做 playback->videoIndex 置换，经 ISC 与标签一致性检验确认属误加，已移除。
    # 播放序仅用于 extract_fea 里的 running_norm（与 FACED After_remarks 流程一致），不在加载期改动数据。

    points_len = int(timeLen * fs)   # 1250
    points_step = int(timeStep * fs) # 500
    n_samples = int((t - timeLen) / timeStep) + 1  # 13
    uniq = sorted(set(int(x) for x in video_labels))
    if min(uniq) < 0 or max(uniq) >= int(n_class):
        raise ValueError(
            f'TY 标签越界: labels={uniq}，n_class={n_class}（来源={label_src}）'
        )
    if subject_video is not None:
        if subject_video.shape[0] < n_subs_actual:
            raise ValueError(
                f'TY label_by_subject_video 被试数={subject_video.shape[0]} < n_subs={n_subs_actual}'
            )
        uniq_sv = sorted(set(int(x) for x in subject_video.ravel().tolist()))
        bad = [x for x in uniq_sv if x != -1 and (x < 0 or x >= int(n_class))]
        if bad:
            raise ValueError(
                f'TY 逐被试标签非法: {bad}，允许 [0, {n_class - 1}] 或 -1(无效剔除)'
            )
        n_invalid = int((subject_video < 0).sum())
        if n_invalid:
            log_io.info('TY per-subject labels: invalid(-1)=%s / %s', n_invalid, subject_video.size)
    log_io.info(
        'TY pkl load: n_subs=%s n_vids=%s n_class=%s label_src=%s per_subject=%s',
        n_subs_actual, n_vids, n_class, label_src, subject_video is not None,
    )

    data = np.empty((n_subs_actual, n_vids * n_samples, n_chans, points_len), dtype=np.float64)
    for idx in range(n_subs_actual):
        sub_data = data_all[idx]  # (n_vids, 31, 7500)
        if sub_data.shape[0] != n_vids:
            raise ValueError(f'TY 被试 {idx} 视频数={sub_data.shape[0]}，期望 {n_vids}')
        if sub_data.shape[1] > n_chans:
            sub_data = sub_data[:, :n_chans, :]  # (n_vids, n_chans, 7500)
        elif sub_data.shape[1] < n_chans:
            raise ValueError(
                f'TY 被试 {idx} 通道数={sub_data.shape[1]}，小于配置 n_chans={n_chans}'
            )
        for vid in range(n_vids):
            for i in range(n_samples):
                start = i * points_step
                end = start + points_len
                data[idx, vid * n_samples + i] = sub_data[vid, :, start:end]
    data = data.reshape(-1, data.shape[-2], data.shape[-1])

    onesub_labels = []
    for lab in video_labels:
        onesub_labels = onesub_labels + [int(lab)] * n_samples
    n_samples_onesub = np.array([n_samples] * n_vids)
    n_samples_sessions = n_samples_onesub.reshape(n_session, -1)
    if subject_video is not None:
        # (n_subs, n_vids*n_samples) — 供 extract_fea 写出 allsubs_label2.npy
        load_processed_TY6_pkl_data.last_allsubs_labels = np.repeat(subject_video, n_samples, axis=1)
    else:
        load_processed_TY6_pkl_data.last_allsubs_labels = None
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions


def pop_last_ty_allsubs_labels():
    """取出最近一次 TY pkl 加载产生的逐被试窗标签；无则返回 None。"""
    labels = getattr(load_processed_TY6_pkl_data, 'last_allsubs_labels', None)
    load_processed_TY6_pkl_data.last_allsubs_labels = None
    return labels


def load_processed_FACED_NEW_data(dir, fs, n_chans, timeLen,timeStep,n_session=1, 
                                  n_subs=123, n_vids = 28, n_class=9, t=30):
    # t: we only use last 30s data for each video
    # input data shape(onesub):(vid,channel,time)
    # output : (subs*slices*vids)*channals*time
    #           (123*28*30)*30*(125)

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    assert len(list_files) == n_subs
    n_samples = int((t-timeLen)/timeStep)+1
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)

    if n_class == 2:
        vid_sel = list(range(12))
        vid_sel.extend(list(range(16,28)))
        # data = data[:, vid_sel, :, :] # sub, vid, n_channs, n_points
        n_vids = 24
    elif n_class == 9:
        vid_sel = list(range(28))
        n_vids = 28
    data = np.empty((n_subs,n_vids*n_samples,n_chans,fs*timeLen),float)
    # subs(*slices*vids)*channals*time

    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        onesubsession_data = sio.loadmat(file_path)
        
        EEG_data = onesubsession_data['data_all_cleaned']
        EEG_data = robust_zscore_per_subject(EEG_data)
        n_points = onesubsession_data['n_samples_one'][0]*fs
        n_points_cum = np.cumsum(n_points).astype(int)
        start_points = n_points_cum-t*fs
        
        for k, vid in enumerate(vid_sel):
            for i in range(n_samples):
                data[idx,k*n_samples+i] = EEG_data[:,start_points[vid]+i*points_step:start_points[vid]+i*points_step+points_len]        
    
    data = data.reshape(-1,data.shape[-2],data.shape[-1])
    # (subs*slices*vids)*channals*time
    
    if n_class == 2:
        label = [0] * 12
        label.extend([1] * 12)
    elif n_class == 9:
        label = [0] * 3
        for i in range(1,4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5,9):
            label.extend([i] * 3)
    
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples
        
    n_samples_onesub = np.array([n_samples]*n_vids)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)

    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions


def load_processed_SEEDV_data(dir, fs, n_chans, timeLen,timeStep, n_session, n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (15*(sum(n_samples_onesub)))*62*point_len(1250)

    list_files = os.listdir(dir)
    list_files.sort(key= lambda x:int(x[:-4]))
    # print(list_files)

    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)


    n_samples_onesub = []
    for i in range(n_session):
        fn = list_files[i]
        file_path = os.path.join(dir,fn)
        onesubsession_data = sio.loadmat(file_path)  
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

    n_samples_sum_onesub = np.sum(n_samples_onesub)


    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    s = np.arange(n_session)
    # n_samples_onesub = []
    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesubsession_data['data']   #(channels,tot_n_points)  (62,tot_n_points)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        # print(EEG_data.shape)
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        
        # if idx < n_session:
        #     if idx == s[idx]:
        #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
        for vid in range(n_vids):
            # print('vid:',vid)
            for i in range(n_samples_onesubsession[vid]):
                # print('sample:',i)

                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1

                # 拼接速度会越来越慢
                # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
                # start_time = time.time()
                # data = np.concatenate((data,temp),0)
                # end_time = time.time()
                # print(end_time - start_time)
    # print(cnt)

    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]
    
    print('load processed data finished!')   

    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions

def load_processed_SEEDV_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3, 
                                  n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # *input data shape（onesub_3session):(channels,tot_time)
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
    

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    assert len(list_files) == n_subs
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)
    
    # 3 session in all change delete the loop
    file_path = os.path.join(dir,list_files[0])
    onesub_data = sio.loadmat(file_path)  
    n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
    n_points = np.array(n_time) * fs
    n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
    n_samples_sum_onesub = np.sum(n_samples_onesub)
    
    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesub_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesub_data['merged_data_all_cleaned']   #(channels,tot_n_points_3session)  (60,tot_n_points_3session)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[np.abs(EEG_data)<thr])) / np.std(EEG_data[np.abs(EEG_data)<thr])
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))

        
        n_vids_all = n_vids*n_session
        for vid in range(n_vids_all):
            # print('vid:',vid)
            for i in range(n_samples_onesub[vid]):
                # print('sample:',i)
                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1
    
    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]   
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions

def load_processed_SEED_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3, 
                                  n_subs=15, n_vids = 15, n_class=3):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # *input data shape（onesub_3session):(channels,tot_time)
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
    

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    assert len(list_files) == n_subs
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)
    
    # 3 session in all change delete the loop
    file_path = os.path.join(dir,list_files[0])
    onesub_data = sio.loadmat(file_path)  
    n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
    n_points = np.array(n_time) * fs
    n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
    n_samples_sum_onesub = np.sum(n_samples_onesub)
    
    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesub_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesub_data['merged_data_all_cleaned']   #(channels,tot_n_points_3session)  (60,tot_n_points_3session)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[np.abs(EEG_data)<thr])) / np.std(EEG_data[np.abs(EEG_data)<thr])
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))

        
        n_vids_all = n_vids*n_session
        for vid in range(n_vids_all):
            # print('vid:',vid)
            for i in range(n_samples_onesub[vid]):
                # print('sample:',i)
                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1
    
    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label =  list(np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])+1) * 3
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]   
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions



def save_sliced_data(sliced_data_dir, data, onesub_labels, n_samples_onesub, n_samples_sessions):
    if not os.path.exists(sliced_data_dir+'/metadata'):
        os.makedirs(sliced_data_dir+'/metadata')
    if not os.path.exists(sliced_data_dir+'/data'):
        os.makedirs(sliced_data_dir+'/data')
    np.save(sliced_data_dir+'/metadata/onesub_labels.npy', onesub_labels)
    np.save(sliced_data_dir+'/metadata/n_samples_onesub.npy', n_samples_onesub)
    np.save(sliced_data_dir+'/metadata/n_samples_sessions.npy', n_samples_sessions)
    for sample in range(data.shape[0]):
        np.save(sliced_data_dir+f'/data/data_sample_{sample}.npy', data[sample])
    np.save(sliced_data_dir+'/saved.npy', [True])
    print('save sliced data finished!')

def test_load_processed_SEEDV_data():
    data_dir = os.environ.get('DAEST_LEGACY_ROOT','./legacy')+'/SEEDV/EEG_processed_sxk'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = os.environ.get('DAEST_LEGACY_ROOT','./legacy')+'/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 250
    n_channs = 62
    n_session = 3

    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_labels)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_labels'] = onesub_labels
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

def test_load_processed_SEEDV_NEW_data():
    data_dir = os.environ.get('DAEST_LEGACY_ROOT','./legacy')+'/SEEDV-NEW/processed_data'
    # data_dir = '.../all/model_weights/grm/SEEDV_new2/processed_ddata'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = os.environ.get('DAEST_LEGACY_ROOT','./legacy')+'/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 125
    n_channs = 60
    n_session = 3

    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_NEW_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_labels)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_labels'] = onesub_labels
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

if __name__ == '__main__':
    # test_load_processed_SEEDV_data()
    test_load_processed_SEEDV_NEW_data()