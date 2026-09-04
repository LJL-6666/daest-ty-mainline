# 原始数据来源

不在本仓库，需从以下位置获取：

| 数据 | 路径 |
|---|---|
| TY 观影 9 类 pkl | `${DAEST_PREP_ROOT}/output/9_movie/data_9class_movie.pkl` |
| TY 讲述 9 类 pkl | `.../output/9_communication/data_9class_communication.pkl` |
| TY 观影 自评 8 类 | `.../output/8_movie_self/data_8class_movie_self.pkl` |
| TY 讲述 自评 8 类 | `.../output/8_communication_self/data_8class_communication_self.pkl` |
| 问卷（播放顺序） | `${DAEST_DATA_ROOT}/data-tongyong/原始数据/问卷/`（129 个被试子目录） |
| FACED 预处理 mat | `${DAEST_DATA_ROOT}/data-faced/0.05–47 Hz_mat/` |

**注意**：`data.questionnaire_dir` 缺失时代码会静默跳过播放序重排（仅打一条 WARNING），
导致口径变成旧高分版。所有 TY 配置的 yaml 中该字段必须存在。
