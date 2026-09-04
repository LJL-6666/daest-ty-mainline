import os
# 无 wandb API key 或 nohup 下用 TensorBoard，避免 wandb 登录报错
_use_wandb = os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes")

import hydra
from omegaconf import DictConfig
from model.models import simpleNN3
import numpy as np
from data.dataset import PDataset
from data.io_utils import get_ty6_save_base, _ty_save_name
from utils.utils import pl_accelerator_devices
from model.pl_models import MLPModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
if _use_wandb:
    import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs

    n_per = round(cfg.data.n_subs / n_folds)
    best_val_acc_list = []
    fold_list = list(range(0, n_folds))
    if getattr(cfg.train, 'fold_index', None) is not None:
        fold_list = [cfg.train.fold_index]
        assert 0 <= fold_list[0] < n_folds, f"train.fold_index={fold_list[0]} 应在 [0, n_folds) 内"
    
    for fold in fold_list:
        ds_name = _ty_save_name(cfg.data)
        cp_dir = os.path.join(cfg.log.cp_dir, ds_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        log_name = cfg.log.exp_name + 'mlp' + 'v' + str(cfg.train.valid_method) + f'_{cfg.data.timeLen}_{cfg.data.timeStep}_r{cfg.log.run}_f{fold}'
        if _use_wandb:
            logger = WandbLogger(name=log_name, project=cfg.log.proj_name, log_model="all")
        else:
            logger = TensorBoardLogger(save_dir=os.path.join(cfg.log.cp_dir, "tb_logs"), name=ds_name, version=log_name)
        cp_monitor = None if n_folds == 1 else "mlp/val/acc"
        es_monitor = "mlp/train/acc" if n_folds == 1 else "mlp/val/acc"
        checkpoint_callback = ModelCheckpoint(monitor=cp_monitor, verbose=True, mode="max", 
                                              dirpath=cp_dir, filename=f'mlp_f{fold}_wd={cfg.mlp.wd}_'+'{epoch}')
        earlyStopping_callback = EarlyStopping(monitor=es_monitor, mode="max", patience=cfg.mlp.patience)
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
        log.info(f'val_subs:{val_subs}')
        
        save_dir = os.path.join(get_ty6_save_base(cfg.data, getattr(cfg.log, 'cp_dir', None)), 'ext_fea', f'fea_r{cfg.log.run}')
        save_path = os.path.join(save_dir,cfg.log.exp_name+'_r'+str(cfg.log.run)+f'_f{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        data2 = np.load(save_path)
        log.info('data2 load from: '+save_path)
        # print(data2[:,160])
        if np.isnan(data2).any():
            log.warning('nan in data2')
            data2 = np.where(np.isnan(data2), 0, data2)
        fea_dim = data2.shape[-1]
        data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-1])
        allsubs_path = os.path.join(save_dir, 'allsubs_label2.npy')
        if os.path.isfile(allsubs_path):
            allsubs_label2 = np.load(allsubs_path)
            if allsubs_label2.shape[0] != cfg.data.n_subs:
                raise ValueError(
                    f'allsubs_label2 n_subs={allsubs_label2.shape[0]} != cfg {cfg.data.n_subs}'
                )
            if allsubs_label2.shape[1] != data2.shape[1]:
                raise ValueError(
                    f'allsubs_label2 n_samples={allsubs_label2.shape[1]} != fea {data2.shape[1]}'
                )
            log.info('using per-subject labels from %s', allsubs_path)
            labels2_train = allsubs_label2[train_subs].reshape(-1)
            labels2_val = allsubs_label2[val_subs].reshape(-1)
        else:
            onesub_label2 = np.load(save_dir+'/onesub_label2.npy')
            labels2_train = np.tile(onesub_label2, len(train_subs))
            labels2_val = np.tile(onesub_label2, len(val_subs))
        trainset2 = PDataset(data2[train_subs].reshape(-1,data2.shape[-1]), labels2_train)
        # trainset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        valset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        trainLoader = DataLoader(trainset2, batch_size=cfg.mlp.batch_size, shuffle=True, num_workers=cfg.mlp.num_workers)
        valLoader = DataLoader(valset2, batch_size=cfg.mlp.batch_size, shuffle=False, num_workers=cfg.mlp.num_workers)
        model_mlp = simpleNN3(fea_dim, cfg.mlp.hidden_dim, cfg.mlp.out_dim,cfg.mlp.dropout,cfg.mlp.bn)
        predictor = MLPModel(model_mlp, cfg.mlp)
        limit_val_batches = 0.0 if n_folds == 1 else 1.0
        acc, dev = pl_accelerator_devices(cfg.mlp.gpus, getattr(cfg.train, 'use_cpu', False))
        trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback, earlyStopping_callback],
                             max_epochs=cfg.mlp.max_epochs, min_epochs=cfg.mlp.min_epochs,
                             accelerator=acc, devices=dev, limit_val_batches=limit_val_batches)
        trainer.fit(predictor, trainLoader, valLoader)
        if cfg.train.valid_method != 1:
            best_val_acc_list.append(checkpoint_callback.best_model_score.item())
        if _use_wandb:
            wandb.finish()
        
        if cfg.train.iftest :
            break
    if cfg.train.valid_method != 1:
        log.info("Best train/validation accuracies for each fold:")
        for fold, acc in enumerate(best_val_acc_list):
            log.info(f"    Fold {fold}: {acc}")
        
        average_val_acc = np.mean(best_val_acc_list)
        log.info(f"Average train/validation accuracy across all folds: {average_val_acc}")
        std_val_acc = np.std(best_val_acc_list)
        log.info(f"Standard deviation of train/validation accuracy across all folds: {std_val_acc}")
        log.info(f"Extracting features with {cfg.mlp.wd}: $mlp_wd and ext_wd: {cfg.train.wd}")

if __name__ == '__main__':
    train_mlp()