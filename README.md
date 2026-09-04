# DAEST · TY 情绪解码 · 主线正确版

跨被试 EEG 情绪解码（DAEST：对比预训练 → 特征提取 → 分类头）。本仓库只收录**主线正确版**（做逐被试稳健归一化 + 做播放序重排 · 双向 LDS）
的代码与结果。旧高分版仅作为方法学风险在 `docs/03_方法学限制.md` 中引用数字，不收录其代码与产物。

## 核心结论

| 实验 | n | 类别 | 随机 | 准确率 | 倍数 |
|---|---|---|---|---|---|
| FACED 素材9 | 123 | 9 | 11.1% | **60.73%** | 5.47× |
| TY 观影 素材9 | 52 | 9 | 11.1% | **30.02%** | 2.70× |
| TY 讲述 素材9 | 118 | 9 | 11.1% | **14.89%** | 1.34× |
| TY 观影 自评8 | 52 | 8 | 12.5% | 26.16% | 2.09× |
| 观影→讲述 同人差 (m51) | 51 | 9 | — | **+13.1 点** | — |

- **观影任务可解码**，约三成；**讲述任务基本做不动**，贴着随机线。
- 冻结 FACED encoder 只重训分类头，观影可达 27.73%，接近本地从头训的 30.02%。
- 全部 16 个实验见 `results/metrics.csv`；主表见 [`docs/00_主表.md`](docs/00_主表.md)，逐项细节见 [`docs/02_实验索引.md`](docs/02_实验索引.md)。
- 准确率均为**被试均值**口径（与报告一致）。总结图：`results/figures/summary/00_总结.png`。

## 目录

```
src/        三步流水线 train_ext → extract_fea → train_mlp，及导出/可视化（21 个文件闭包）
scripts/    00–40 复现入口 + build_metrics/build_figures/build_docs_index/verify_repo
weights/    encoder 32 个 + 分类头 40 个（60 MB）——有它即可跳过 train_ext
results/    metrics.csv（★机器可读主表）、视频级 OOF、图（分五类）、摘要 JSON
docs/       00 主表 · 01 结论与方法学依据 · 02 实验索引 · 03 方法学限制 · 04 讲稿
            （00 与 02 由 metrics.csv 自动生成，勿手工编辑）
env/        environment.yml（实测版本）· PATHS.md（环境变量）· VERSIONS.md（资源注意事项）
MANIFEST/   大文件清单（4.8 GB 未入库）与原始数据来源
```

**从哪读起**：`docs/00_主表.md` 看结果 → `docs/01` 看为什么这么设置 →
`docs/03` 看限制 → `results/figures/per_experiment/` 看每个实验的图。

## 关于粒度

因双向 LDS 按视频平滑，**视频内预测恒定率 97–99%**，有效样本单位是 **28 个视频**而非 840 个窗口。
故 `results/oof_video_level/` 收录**视频级** OOF（每实验约 31 KB）——这不是压缩，是本来就该报的粒度。
窗口级 OOF（共 213 MB）与特征（4.6 GB）不入库，见 `MANIFEST/artifacts.tsv`。

## 复现：能做到哪一步

**克隆后立刻可做（不需要任何外部数据）：**

```bash
python scripts/verify_repo.py     # 自检：视频级 OOF 重算 vs metrics.csv、塌陷标注、代码可导入
```

- 读 `results/metrics.csv`（16 个实验的全部指标）与 `results/oof_video_level/`（视频级预测）
- 由视频级 OOF **独立重算**每个实验的准确率并与表核对（自检脚本已做，误差 < 0.05 点）
- 看 `results/figures/`（五类子目录，见其 README）全部图，读 `docs/`

**需要 encoder 权重（已含在 `weights/`，60 MB）：**

```bash
conda env create -f env/environment.yml && conda activate daest
source scripts/00_check_env.sh      # 逐块验 GPU，自动跳过故障卡
# 跳过 train_ext，直接提特征 → 训分类头
```

**需要原始 EEG 数据（不在本仓库，见 `MANIFEST/data_sources.md`）：**

从零跑通 `scripts/10_baselines.sh` 等四个脚本。数据需另行获取授权。

### 诚实说明

| 项 | 状态 |
|---|---|
| 结果可核对 | ✅ 视频级 OOF 与 metrics.csv 自洽，自检脚本可验 |
| 代码可导入、无绝对路径 | ✅ 全部走环境变量，见 `env/PATHS.md` |
| encoder 权重齐备 | ✅ 5 套，可跳过对比预训练 |
| 逐窗 OOF 与特征 | ❌ 共 4.8 GB，未入库，见 `MANIFEST/artifacts.tsv`（可由权重重算） |
| **原始 EEG 数据** | ❌ **不在本仓库，且需授权**——因此外部无法从零完整复现 |
| `scripts/` 四个入口 | ⚠ 已写但**未在干净环境端到端验证过**；目录命名沿用历史约定，首次运行可能需按实际路径调整 |

## ⚠ 使用本仓库结果前必读

`docs/03_方法学限制.md` 列了六条限制，其中三条会影响结论的表述方式：
早停用验证折本身、观影线全部建立在与 FACED 相同的刺激上、`running_norm` 关闭后分类器塌陷。
