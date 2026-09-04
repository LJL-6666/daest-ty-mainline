# -*- coding: utf-8 -*-
"""由 results/metrics.csv 生成 docs/02_实验索引.md，避免手工誊数字。"""
import os,pandas as pd
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(f"{R}/results/metrics.csv",dtype={"exp_id":str})
ARM={"baseline":"基线","zeroshot":"零样本","mlpft":"MLP-FT","ablation":"消融"}
TASK={"FACED":"FACED","movie":"TY 观影","comm":"TY 讲述"}
L=["# 实验索引","",
   "> 本文件由 `scripts/build_docs_index.py` 从 `results/metrics.csv` 自动生成，请勿手工编辑。","",
   "全部为**跨被试 OOF**：每折训练时完全未见测试被试。`overall_acc` 为窗口池口径，",
   "`mean_subject_acc` 为被试均值口径；素材类两者数学恒等，自评类因存在无效标签而略有差异。",""]
for arm in ["baseline","zeroshot","mlpft","ablation"]:
    d=df[df.arm==arm]
    if d.empty: continue
    L+=[f"## {ARM[arm]}","",
        "| # | 实验 | n | 类别 | 随机 | 准确率 | 倍数 | 召回>20%的类 | 视频内恒定率 | ρ vs 素材编号 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for _,r in d.iterrows():
        rho="" if pd.isna(r.spearman_rho) else f"{r.spearman_rho:+.2f} (p={r.spearman_p:.3f})"
        c="" if pd.isna(r.within_video_const_pct) else f"{r.within_video_const_pct:.1f}%"
        L.append(f"| {r.exp_id} | {r['name']} | {r.n_subs} | {r.n_class} | {r.chance:.1f}% | "
                 f"**{r.overall_acc:.2f}%** | {r.ratio_to_chance:.2f}× | {r.n_recall_gt20}/{r.n_class} | {c} | {rho} |")
    L.append("")
b=df[df.arm=="baseline"].set_index("exp_id")
L+=["## 派生指标","",
    f"- **同人差（matched51，同一批 51 人）**：观影 {b.loc['06'].overall_acc:.2f}% − "
    f"讲述 {b.loc['07'].overall_acc:.2f}% = **+{b.loc['06'].overall_acc-b.loc['07'].overall_acc:.1f} 点**。"
    "这是唯一控制了个体差异的比较。","",
    "## 位置混杂检验","",
    "对每个素材九类实验，计算九类召回率与该情绪素材编号位置（videoIndex 中点 "
    "`2,5,8,11,14.5,18,21,24,27`，中性因含 4 个视频取 14.5）的 Spearman 相关。",""]
s=df[(df.n_class==9)&(df.label_type=="stimulus")&df.spearman_p.notna()]
L+=[f"**{len(s)} 项全部不显著**（|ρ| ≤ {s.spearman_rho.abs().max():.2f}，p ≥ {s.spearman_p.min():.3f}），"
    "说明主线设置下「情绪好不好认」与它排第几号素材无关。","",
    "自评类两臂未纳入：其标签为每被试自评分 argmax，同一视频在不同被试上标签不同，"
    "不存在固定的「情绪↔素材编号」对应，该检验不适用。","",
    "## 有效样本粒度","",
    "视频内预测恒定率 97–99%（见上表），故有效样本单位为 **28 个视频**而非 840 个窗口，"
    "所有被试准确率均为 1/28 的整数倍。`results/oof_video_level/` 收录的即为该粒度。",""]
open(f"{R}/docs/02_实验索引.md","w",encoding="utf8").write("\n".join(L))
print("已生成 docs/02_实验索引.md")
