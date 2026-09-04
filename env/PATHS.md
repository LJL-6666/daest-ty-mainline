# 路径配置

本仓库不含任何绝对路径。运行前需设置以下环境变量（按需，只有实际用到的才必须设）：

| 变量 | 含义 | 示例 |
|---|---|---|
| `DAEST_RUNS_ROOT` | 实验产物根目录（特征、权重、OOF） | `/path/to/FACED-base` |
| `DAEST_PREP_ROOT` | TY 预处理输出根目录（各 `.pkl`） | `/path/to/Preprocessing/全部的TY` |
| `DAEST_DATA_ROOT` | 原始数据根（问卷 CSV、FACED mat） | `/path/to/TY/data` |
| `DAEST_RIEM_ROOT` | FACED r3 特征所在（仅 FACED preset 用），默认同 `DAEST_RUNS_ROOT` | — |
| `DAEST_LEGACY_ROOT` | SEED / 旧 FACED 的遗留默认值，本项目不用 | 可不设 |

```bash
export DAEST_RUNS_ROOT=/path/to/FACED-base
export DAEST_PREP_ROOT=/path/to/Preprocessing/全部的TY
export DAEST_DATA_ROOT=/path/to/TY/data
```

Hydra 配置中用 `${oc.env:VAR}` 引用，Python 中用 `os.environ.get("VAR", 默认)`。
