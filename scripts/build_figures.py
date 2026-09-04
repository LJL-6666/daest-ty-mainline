# -*- coding: utf-8 -*-
"""为 metrics.csv 中每个实验统一生成「逐被试准确率条形图 + 混淆矩阵」双联图。
   塌陷的实验（单类召回>80% 且多数类接近 0）会在标题上红字标注。"""
import os,glob,numpy as np,pandas as pd,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Noto Sans CJK JP","Noto Sans CJK SC","WenQuanYi Zen Hei"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist): plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC=os.environ.get("DAEST_RUNS_ROOT",".")
C9=['愤怒','厌恶','恐惧','悲伤','中性','愉悦','鼓舞','喜悦','温柔']
C8=['愤怒','厌恶','恐惧','悲伤','愉悦','鼓舞','喜悦','温柔']
OUT=f"{R}/results/figures/per_experiment"; os.makedirs(OUT,exist_ok=True)
df=pd.read_csv(f"{R}/results/metrics.csv",dtype={"exp_id":str})
for _,m in df.iterrows():
    g=glob.glob(os.path.join(SRC,m.oof_source))
    if not g: print(f"  ⚠ {m.exp_id} 缺 OOF"); continue
    cols=pd.read_csv(g[0],nrows=0).columns.tolist()
    use=[c for c in ['y_true','y_pred','valid_for_metric','subject_index'] if c in cols]
    d=pd.read_csv(g[0],usecols=use)
    if 'valid_for_metric' in d: d=d[d.valid_for_metric==1]
    d=d[d.y_true>=0]
    K=int(m.n_class); names=C9 if K==9 else C8; ch=100/K
    acc=100*(d.y_pred==d.y_true).mean()
    sub=d.groupby('subject_index').apply(lambda x:100*(x.y_pred==x.y_true).mean(),include_groups=False)
    rec=np.array([float(x) for x in m.recall_by_class.split("|")])
    degen = rec.max()>80 and (rec>20).sum()<=max(1,K//4)
    fig,ax=plt.subplots(1,2,figsize=(15,5.6),gridspec_kw=dict(width_ratios=[1.25,1]))
    v=np.sort(sub.values)
    ax[0].bar(range(len(v)),v,color='#c9c9c9',edgecolor='#8a8a8a',lw=.3,width=.9)
    ax[0].bar([len(v)+3],[acc],color='#7ddc7d',edgecolor='#3a8f3a',lw=.8,width=2.5)
    ax[0].errorbar([len(v)+3],[acc],yerr=[sub.std()],color='k',capsize=4,lw=1.3)
    ax[0].text(len(v)+3,acc+sub.std()+1.5,f"{acc:.2f}%",ha='center',fontsize=11,fontweight='bold')
    ax[0].axhline(ch,color='r',ls='--',lw=1.5,label=f'随机 {ch:.1f}%')
    ax[0].set_xlim(-2,len(v)+7); ax[0].set_ylabel("准确率 (%)"); ax[0].set_xlabel("被试（按准确率升序）· 最右为均值")
    ax[0].legend(loc='upper left',fontsize=9); ax[0].grid(axis='y',alpha=.25)
    ax[0].set_title(f"逐被试准确率 · SD={sub.std():.2f} · 范围 {v.min():.1f}–{v.max():.1f}%",fontsize=11)
    cm=np.zeros((K,K))
    for t in range(K):
        msk=d.y_true==t; n=max(msk.sum(),1)
        for p in range(K): cm[t,p]=100*((d.y_pred==p)&msk).sum()/n
    im=ax[1].imshow(cm,cmap='Blues',vmin=0,vmax=max(35,cm.max()))
    for t in range(K):
        for p in range(K):
            ax[1].text(p,t,f"{cm[t,p]:.0f}",ha='center',va='center',fontsize=8.5,
                       color='white' if cm[t,p]>cm.max()*.6 else '#333',
                       fontweight='bold' if t==p else 'normal')
    ax[1].set_xticks(range(K)); ax[1].set_xticklabels(names,rotation=45,ha='right',fontsize=9)
    ax[1].set_yticks(range(K)); ax[1].set_yticklabels(names,fontsize=9)
    ax[1].set_xlabel("预测"); ax[1].set_ylabel("真实")
    ax[1].set_title(f"混淆矩阵（行归一化 %）· 对角线宏平均 {rec.mean():.1f}%",fontsize=11)
    plt.colorbar(im,ax=ax[1],fraction=.046,label='%')
    t=f"{m.exp_id} · {m['name']}   n={m.n_subs} · {K} 类 · 随机 {ch:.1f}% · 总体 {acc:.2f}% ({m.ratio_to_chance:.2f}×)"
    if degen: t+="\n⚠ 分类器塌陷：预测集中于单一类别，该准确率不代表有效性能"
    fig.suptitle(t,fontsize=13,fontweight='bold',color=('#B00' if degen else 'black'),y=1.0)
    fig.tight_layout(rect=[0,0,1,0.94 if degen else 0.97])
    fn=f"{OUT}/exp{m.exp_id}_{m.task}_{m.label_type}_{K}class.png"
    fig.savefig(fn,dpi=140,bbox_inches='tight'); plt.close(fig)
    print(f"  {m.exp_id} {'⚠塌陷 ' if degen else '     '}{os.path.basename(fn)}")
print(f"\n输出至 results/figures/per_experiment/")
