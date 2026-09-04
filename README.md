# DAEST · TY 情绪解码 · 主线正确版

跨被试 EEG 情绪解码。本仓库只收录**主线正确版**（做逐被试稳健归一化 + 做播放序重排 · 双向 LDS）
的代码与结果。旧高分版仅作为方法学风险在 `docs/03_方法学限制.md` 中引用数字，不收录其代码与产物。

## 核心结论

| 实验 | n | 类别 | 随机 | 准确率 | 倍数 |
|---|---|---|---|---|---|
| FACED 素材9 | 123 | 9 | 11.1% | **60.73%** | 5.47× |
| TY 观影 素材9 | 52 | 9 | 11.1% | **30.02%** | 2.70× |
| TY 讲述 素材9 | 118 | 9 | 11.1% | **14.89%** | 1.34× |
| TY 观影 自评8 | 52 | 8 | 12.5% | 26.09% | 2.09× |
| 观影→讲述 同人差 (m51) | 51 | 9 | — | **+13.1 点** | — |

- **观影任务可解码**，约三成；**讲述任务基本做不动**，贴着随机线。
- 冻结 FACED encoder 只重训分类头，观影可达 27.73%，接近本地从头训的 30.02%。
- 全部 16 个实验见 `results/metrics.csv`。

## 目录

```
src/        三步流水线 train_ext → extract_fea → train_mlp，及导出/可视化
scripts/    可复现入口 + build_metrics.py
results/    metrics.csv（★机器可读主表）、视频级 OOF、图、摘要 JSON
docs/       结论与证据链、实验索引、方法学限制、讲稿
MANIFEST/   大文件清单与原始数据来源
```

## 关于粒度

因双向 LDS 按视频平滑，**视频内预测恒定率 97–99%**，有效样本单位是 **28 个视频**而非 840 个窗口。
故 `results/oof_video_level/` 收录**视频级** OOF（每实验约 31 KB）——这不是压缩，是本来就该报的粒度。
窗口级 OOF（共 213 MB）与特征（4.6 GB）不入库，见 `MANIFEST/artifacts.tsv`。

## 复现

```bash
conda env create -f env/environment.yml && conda activate daest
bash scripts/00_check_env.sh          # 验环境 + 逐块验 GPU（带超时，自动跳过故障卡）
bash scripts/10_baselines.sh          # 实验 01–07
bash scripts/20_transfers.sh          # 实验 08–13
bash scripts/30_ablations.sh          # 实验 30–32
bash scripts/40_export_figures.sh     # OOF 导出 + metrics.csv + 全部图
```

原始数据不在本仓库。先按 `env/PATHS.md` 设置环境变量，数据来源见 `MANIFEST/data_sources.md`。
有 encoder 权重（36 MB，git-lfs）即可跳过 `train_ext`，直接从 `extract_fea` 起跑。

## ⚠ 使用本仓库结果前必读

`docs/03_方法学限制.md` 列了六条限制，其中三条会影响结论的表述方式：
早停用验证折本身、观影线全部建立在与 FACED 相同的刺激上、`running_norm` 关闭后分类器塌陷。
