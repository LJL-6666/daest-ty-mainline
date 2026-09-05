# 图

| 目录 | 内容 | 来源 |
|---|---|---|
| `summary/` | 总结图、一页总结、迁移对照 | 汇总视图，每个结果只出现一次 |
| **`per_experiment/`** | **16 个实验各一张双联图**（左：逐被试准确率；右：混淆矩阵） | `scripts/build_figures.py` 自动生成，**格式统一、可重现，以此为准** |
| `dynamics/` | 逐秒动态热图（LDS 三档对比、自评与 m51、FACED 每人每秒） | 时间维度分析 |

**分类器塌陷的实验**在 `per_experiment/` 的图上会有红字标注
（判据：某类召回 >80% 且过 20% 的类不足 `n_class/4`，见 `results/metrics.csv` 的 `status` 列）。
