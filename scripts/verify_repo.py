# -*- coding: utf-8 -*-
"""仓库自检：只用仓库内文件，不依赖原始数据与大文件。
   校验视频级 OOF 能否重算出 metrics.csv 中的准确率，并检查引用完整性。"""
import os,sys,glob,numpy as np,pandas as pd
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok=True
def chk(cond,msg):
    global ok; print(("  ✅ " if cond else "  ❌ ")+msg); ok = ok and cond

print("=== 1. 结果文件完整性 ===")
m=f"{R}/results/metrics.csv"; chk(os.path.isfile(m),"results/metrics.csv 存在")
df=pd.read_csv(m,dtype={"exp_id":str})
chk(len(df)==16,f"metrics.csv 有 16 行（实际 {len(df)}）")
for c in ["exp_id","overall_acc","chance","n_class","status","recall_by_class"]:
    chk(c in df.columns,f"含列 {c}")

print("\n=== 2. 视频级 OOF 能否重算出 metrics.csv 的准确率 ===")
nchk=0
for _,r in df.iterrows():
    g=glob.glob(f"{R}/results/oof_video_level/exp{r.exp_id}_*.csv")
    if not g: continue
    d=pd.read_csv(g[0]); a=100*(d.y_pred==d.y_true).mean()
    diff=abs(a-r.overall_acc)
    chk(diff<0.6,f"{r.exp_id} 视频级重算 {a:.2f}% vs 表中 {r.overall_acc:.2f}%（差 {diff:.2f}）")
    nchk+=1
print(f"  （零样本三项无 video_index，不含视频级 OOF；已校验 {nchk} 项）")

print("\n=== 3. 塌陷标注是否与逐类召回一致 ===")
for _,r in df.iterrows():
    rec=np.array([float(x) for x in r.recall_by_class.split("|")])
    exp = rec.max()>80 and (rec>20).sum()<=max(1,int(r.n_class)//4)
    chk((r.status=="degenerate")==exp, f"{r.exp_id} status={r.status}（最大召回 {rec.max():.1f}%，过20%的类 {(rec>20).sum()}/{r.n_class}）")

print("\n=== 4. 代码可导入 ===")
sys.path.insert(0,f"{R}/src")
for mod in ["data.io_utils","data.dataset","model.models","model.pl_models","utils.reorder_vids"]:
    try: __import__(mod); chk(True,f"import {mod}")
    except Exception as e: chk(False,f"import {mod} → {type(e).__name__}")

print("\n=== 5. 权重 ===")
ne=len(glob.glob(f"{R}/weights/encoders/**/*.ckpt",recursive=True))
nc=len(glob.glob(f"{R}/weights/classifiers/**/*.ckpt",recursive=True))
chk(ne>0,f"encoder {ne} 个"); chk(nc>0,f"分类头 {nc} 个")

print("\n=== 6. 文档引用的图是否存在 ===")
import re
miss=[]
for f in glob.glob(f"{R}/docs/*.md")+[f"{R}/README.md"]:
    for p in re.findall(r'\]\((\.\./results/figures/[^)]+)\)',open(f,encoding='utf8').read()):
        t=os.path.normpath(os.path.join(os.path.dirname(f),p))
        if not os.path.exists(t): miss.append(p)
chk(not miss, f"文档引用的图全部存在" + (f"（缺 {miss}）" if miss else ""))

print("\n" + ("="*50) + f"\n{'全部通过 ✅' if ok else '存在失败项 ❌'}")
sys.exit(0 if ok else 1)
