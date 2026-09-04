# 权重

| | 内容 | 说明 |
|---|---|---|
| `encoders/` | `train_ext` 产出的对比预训练 encoder，`f<fold>epoch=*.ckpt` | **有它即可跳过 `train_ext`** |
| `classifiers/` | `train_mlp` 产出的分类头，`mlp_f<fold>_wd=*.ckpt` | 有它 + 特征即可直接复现 OOF |

## 五套 encoder，七个实验共用

| encoder | 供哪些实验使用 |
|---|---|
| `FACED_05_47_cp` | 01 FACED；08/09 零样本源；11/12 MLP-FT 源 |
| `TY9_movie_FACEDalign_cp` | 02 观影素材9；**04 观影自评8**（对比预训练不用情绪标签，故同一 encoder 可服务两种标签） |
| `TY9_communication_FACEDalign_cp` | 03 讲述素材9；05 讲述自评8；31/32 消融 |
| `TY9_movie_matched51_FACEDalign_cp` | 06 观影 m51；10/13 迁移源 |
| `TY9_communication_matched51_FACEDalign_cp` | 07 讲述 m51 |

## 用法

```bash
python src/extract_fea.py data=TY9_movie log.run=1 \
  "+ext_fea.load_cp_dir=$PWD/weights/encoders/TY9_movie_FACEDalign_cp" \
  ext_fea.ckpt_run=1 ext_fea.mode=me \
  ext_fea.use_lds=true ext_fea.lds_given_all=1 \
  train.valid_method=5 train.fold_index=0
```

`ckpt_dataset` 默认取配置的 `save_dataset_name`；跨臂复用时需显式指定，
例如自评臂用观影 encoder：`+ext_fea.ckpt_dataset=TY9_movie`。
