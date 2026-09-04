import os
# 无 wandb API key 或 nohup 下用 TensorBoard，避免 wandb 登录报错
_use_wandb = os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes")

import hydra
from omegaconf import DictConfig
import torch
from model import Conv_att_simple_new, ExtractorModel
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
if _use_wandb:
    import wandb
from data.pl_datamodule import EEGDataModule,SEEDVDataModule, FACEDDataModule
from data.io_utils import _ty_save_name
from utils.utils import pl_accelerator_devices

# normalize data


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def train_ext(cfg: DictConfig) -> None:
    # set logger
   
    # set seed
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
        print("fold:", fold)
        ds_name = _ty_save_name(cfg.data)
        cp_dir = os.path.join(cfg.log.cp_dir, ds_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        log_name = cfg.log.exp_name + 'v' + str(cfg.train.valid_method) + f'_{cfg.data.timeLen}_{cfg.data.timeStep}_r{cfg.log.run}_f{fold}'
        if _use_wandb:
            logger = WandbLogger(name=log_name, project=cfg.log.proj_name, log_model="all")
        else:
            logger = TensorBoardLogger(save_dir=os.path.join(cfg.log.cp_dir, "tb_logs"), name=ds_name, version=log_name)
        checkpoint_callback = ModelCheckpoint(monitor="ext/val/acc", mode="max", dirpath=cp_dir, filename=f'f{fold}'+'{epoch}')
        earlyStopping_callback = EarlyStopping(monitor="ext/val/acc", mode="max", patience=cfg.train.patience)
        # split data
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        if len(val_subs) == 1:
            val_subs = list(val_subs) + train_subs
        print('train_subs:', train_subs)
        print('val_subs:', val_subs)
        
        # load_dir = os.path.join(cfg.data.data_dir,'processed_data')
        # save_dir = os.path.join(cfg.data.data_dir,'sliced_data')
        if cfg.data.dataset_name == 'FACED':
            if cfg.data.n_class == 2:
                n_vids = 24
            elif cfg.data.n_class == 9:
                n_vids = 28
        elif cfg.data.dataset_name == 'TY6':
            # 6 类=19 视频 / 9 类=28 视频，一律以配置为准，避免写死 19 误伤 TY9
            n_vids = int(cfg.data.n_vids)
        else:
            n_vids = cfg.data.n_vids
        train_vids = np.arange(n_vids)
        val_vids = np.arange(n_vids)
        # if cfg.data.dataset_name == 'SEEDV':
        #     dm = SEEDVDataModule(load_dir, save_dir, cfg.data.timeLen, cfg.data.timeStep, train_subs,
        #                          val_subs, train_vids, val_vids, cfg.data.n_session, cfg.data.fs, 
        #                          cfg.data.n_channs, cfg.data.n_subs, cfg.data.n_vids, cfg.data.n_class,
        #                          cfg.train.valid_method=='loo', cfg.train.num_workers)
        # elif cfg.data.dataset_name == 'FACED':
        #     dm = FACEDDataModule(load_dir, save_dir, cfg.data.timeLen, cfg.data.timeStep, train_subs,
        #                          val_subs, train_vids, val_vids, cfg.data.n_session, cfg.data.fs, 
        #                          cfg.data.n_channs, cfg.data.n_subs, cfg.data.n_vids, cfg.data.n_class,
        #                          cfg.train.valid_method=='loo', cfg.train.num_workers)
        # elif cfg.data.dataset_name == 'SEED':
        #     pass
        dm = EEGDataModule(cfg.data, train_subs, val_subs, train_vids, val_vids,
                           cfg.train.valid_method=='loo', cfg.train.num_workers)
            

        # load model
        model = Conv_att_simple_new(cfg.model.n_timeFilters, cfg.model.timeFilterLen, cfg.model.n_msFilters, cfg.model.msFilterLen, cfg.model.n_channs, cfg.model.dilation_array, cfg.model.seg_att, 
            cfg.model.avgPoolLen, cfg.model.timeSmootherLen, cfg.model.multiFact, cfg.model.stratified,  cfg.model.activ, cfg.model.ext_temp, cfg.model.saveFea, cfg.model.has_att, cfg.model.extract_mode, cfg.model.global_att)
        
        Extractor = ExtractorModel(model, cfg.train)
        acc, dev = pl_accelerator_devices(cfg.train.gpus, getattr(cfg.train, 'use_cpu', False))
        trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback, earlyStopping_callback], max_epochs=cfg.train.max_epochs, min_epochs=cfg.train.min_epochs, accelerator=acc, devices=dev)
        trainer.fit(Extractor, dm)
        if _use_wandb:
            wandb.finish()
        
        if cfg.train.iftest :
            break


    


if __name__ == "__main__":
    train_ext()
