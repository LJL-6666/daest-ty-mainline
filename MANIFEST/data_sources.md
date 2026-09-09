# 原始数据来源

## 上游预处理仓库

本仓库表格中列出的 pkl / mat **全部由预处理仓库产出**：

**https://github.com/LJL-6666/daest-ty-preprocessing**

两仓库通过同一组环境变量对接：`DAEST_DATA_ROOT`（原始数据根目录）、
`DAEST_PREP_ROOT`（预处理仓库产出根目录）。上游两份文档（链接指向上游仓库，非本仓库）：

- [与建模仓库的衔接](https://github.com/LJL-6666/daest-ty-preprocessing/blob/main/docs/02_%E4%B8%8E%E5%BB%BA%E6%A8%A1%E4%BB%93%E5%BA%93%E7%9A%84%E8%A1%94%E6%8E%A5.md)：逐个 yaml 字段的对应关系
- [版本溯源](https://github.com/LJL-6666/daest-ty-preprocessing/blob/main/docs/04_%E7%89%88%E6%9C%AC%E6%BA%AF%E6%BA%90.md)：哪一份预处理代码产出了哪一份数据（以 md5 / mtime / 产出时间三重证据固定）

| 数据集 | 上游链路 |
|---|---|
| TY | `.bdf` → `main_tongyong.py` + `Preprocessing_tongyong.py` → 逐被试 pkl → `extract_{9class,8class}_*.py` → 汇总 pkl |
| FACED | `.bdf` → `main_faced.py` + `Preprocessing.py`（MNE，**不需要 MATLAB/EEGLAB**）→ 逐被试 pkl → `convert_pkl_to_mat.py` → `.mat` |

口径：两个数据集均为 250 Hz · 28 视频 × 30 秒 · 0.05–47 Hz 带通；
TY 31 通道 / FACED 32 通道。

## 数据文件

不在本仓库，需从以下位置获取：

| 数据 | 路径 |
|---|---|
| TY 观影 9 类 pkl | `${DAEST_PREP_ROOT}/output/9_movie/data_9class_movie.pkl` |
| TY 讲述 9 类 pkl | `.../output/9_communication/data_9class_communication.pkl` |
| TY 观影 自评 8 类 | `.../output/8_movie_self/data_8class_movie_self.pkl` |
| TY 讲述 自评 8 类 | `.../output/8_communication_self/data_8class_communication_self.pkl` |
| 问卷（播放顺序） | `${DAEST_DATA_ROOT}/data-tongyong/原始数据/问卷/`（129 个被试子目录） |
| TY 观影 matched51 | `.../output/9_movie_matched51/data_9class_movie_matched51.pkl` |
| TY 讲述 matched51 | `.../output/9_communication_matched51/data_9class_communication_matched51.pkl` |
| FACED 预处理 mat | `${DAEST_DATA_ROOT}/data-faced/0.05–47 Hz_mat/processed_data/subXXX.mat`（123 人） |
| FACED 播放顺序 | `${DAEST_DATA_ROOT}/data-faced/Data/subXXX/After_remarks.mat` |

**注意**：`data.questionnaire_dir` 缺失时代码会静默跳过播放序重排（仅打一条 WARNING），
导致口径变成旧高分版。所有 TY 配置的 yaml 中该字段必须存在。

**注意**：`matched51` 子集（双任务共同 51 人）的生成脚本已遗失，但其判据完整保存在
上游仓库的 [`results/qc_summaries/`](https://github.com/LJL-6666/daest-ty-preprocessing/blob/main/results/qc_summaries)（两份
`9_*_matched51_qc_summary.json`，含 51 个 `subject_ids`、排除的被试 `079`、
`data_shape`），可据此复现。

**注意**：FACED 预处理的 ICA 未固定随机种子，重跑得到的成分次序可能与本仓库结果
所用的数据不同。复现请以已产出的 `.mat` 为准，而非重新预处理。
