# -*- coding: utf-8 -*-
"""从各实验的逐窗 OOF 生成 results/metrics.csv 与 results/oof_video_level/。
   正文所有表格与森林图应由本文件产出的 metrics.csv 驱动，不要手工誊数字。"""
import glob,os,numpy as np,pandas as pd
from scipy.stats import spearmanr
SRC=os.environ.get("DAEST_RUNS_ROOT","{RUNS_ROOT}")
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS9=[2,5,8,11,14.5,18,21,24,27]   # 九类情绪的 videoIndex 中点（中性 4 个视频 → 14.5）
E=[
 ("01","FACED 素材9","baseline","FACED","stimulus",123,9,10,"on","runs/versionA_FACED_biLDS_cp/runs/window_predictions_oof_r*/*window_predictions_oof_r*.csv"),
 ("02","TY 观影 素材9","baseline","movie","stimulus",52,9,5,"on","runs/TY9_movie_mainline_biLDS_cp/runs/window_predictions_oof_r*/*window_predictions_oof_r*.csv"),
 ("03","TY 讲述 素材9","baseline","comm","stimulus",118,9,5,"on","runs/TY9_communication_mainline_biLDS_cp/runs/window_predictions_oof_r*/*window_predictions_oof_r*.csv"),
 ("04","TY 观影 自评8","baseline","movie","selfreport",52,8,5,"on","runs/TY8_movie_self_mainline_biLDS_cp/runs/window_predictions_oof_r*/*window_predictions_oof_r*.csv"),
 ("05","TY 讲述 自评8","baseline","comm","selfreport",116,8,5,"on","runs/TY8_communication_self_mainline_biLDS_cp/runs/window_predictions_oof_r*/*window_predictions_oof_r*.csv"),
 ("06","观影 matched51","baseline","movie","stimulus",51,9,5,"on","runs/TY9_movie_matched51_mainline_biLDS_cp/runs/window_predictions_oof_r*/*window_predictions_oof_r*.csv"),
 ("07","讲述 matched51","baseline","comm","stimulus",51,9,5,"on","runs/TY9_communication_matched51_mainline_biLDS_cp/runs/window_predictions_oof_r*/*window_predictions_oof_r*.csv"),
 ("08","零样本 FACED→观影","zeroshot","movie","stimulus",52,9,10,"on","runs/versionA_biLDS_transfer/FACED_to_TY9_movie_zeroshot_biLDSmlp_cp/runs/zeroshot_eval/window_predictions_ensemble.csv"),
 ("09","零样本 FACED→讲述","zeroshot","comm","stimulus",118,9,10,"on","runs/versionA_biLDS_transfer/FACED_to_TY9_communication_zeroshot_biLDSmlp_cp/runs/zeroshot_eval/window_predictions_ensemble.csv"),
 ("10","零样本 观影→讲述m51","zeroshot","comm","stimulus",51,9,5,"on","runs/versionA_biLDS_transfer/TY9_movie_to_TY9_communication_zeroshot_matched51_biLDSmlp_cp/runs/zeroshot_eval/window_predictions_ensemble.csv"),
 ("11","MLP-FT FACED→观影","mlpft","movie","stimulus",52,9,5,"on","runs/DAEST_revised_transfer_queue/FACED_to_TY9_movie_mlp_ft_revised_cp/runs/window_predictions_oof_r1/*window_predictions_oof_r1.csv"),
 ("12","MLP-FT FACED→讲述","mlpft","comm","stimulus",118,9,5,"on","runs/DAEST_revised_transfer_queue/FACED_to_TY9_communication_mlp_ft_revised_cp/runs/window_predictions_oof_r1/*window_predictions_oof_r1.csv"),
 ("13","MLP-FT 观影→讲述m51","mlpft","comm","stimulus",51,9,5,"on","runs/DAEST_revised_transfer_queue/TY9_movie_to_TY9_communication_mlp_ft_matched51_revised_cp/runs/window_predictions_oof_r1/*window_predictions_oof_r1.csv"),
 ("30","消融 讲述素材8 去中性","ablation","comm","stimulus",118,8,5,"on","runs/TY9_communication_8class_A_cp/runs/window_predictions_oof_r1/*window_predictions_oof_r*.csv"),
 ("31","消融 讲述9 关 running_norm","ablation","comm","stimulus",118,9,5,"off","runs/TY9_communication_noRN_A_cp/runs/window_predictions_oof_r904/*window_predictions_oof_r904.csv"),
 ("32","消融 讲述8 关 running_norm","ablation","comm","stimulus",118,8,5,"off","runs/TY9_communication_noRN_8class_A_cp/runs/window_predictions_oof_r904/*window_predictions_oof_r904.csv"),
]
rows=[]; os.makedirs(f"{REPO}/results/oof_video_level",exist_ok=True)
for eid,name,arm,task,lab,ns,nc,nf,rn,pat in E:
    g=glob.glob(os.path.join(SRC,pat))
    if not g: print(f"  ⚠ {eid} {name}: 未找到 OOF"); continue
    f=g[0]; cols=pd.read_csv(f,nrows=0).columns.tolist()
    use=[c for c in ['y_true','y_pred','valid_for_metric','subject_index','video_index'] if c in cols]
    d=pd.read_csv(f,usecols=use)
    if 'valid_for_metric' in d: d=d[d.valid_for_metric==1]
    d=d[d.y_true>=0]
    acc=100*(d.y_pred==d.y_true).mean()
    sub=d.groupby('subject_index').apply(lambda x:100*(x.y_pred==x.y_true).mean(),include_groups=False)
    rec=[100*((d.y_pred==k)&(d.y_true==k)).sum()/max((d.y_true==k).sum(),1) for k in range(nc)]
    const=100*(d.groupby(['subject_index','video_index'])['y_pred'].nunique()==1).mean() if 'video_index' in d else np.nan
    rho=p=np.nan
    if nc==9 and lab=="stimulus": rho,p=map(float,spearmanr(POS9,rec))
    if 'video_index' in d:
        v=(d.groupby(['subject_index','video_index'])
             .agg(y_true=('y_true','first'),y_pred=('y_pred',lambda s:s.mode().iloc[0])).reset_index())
        v.to_csv(f"{REPO}/results/oof_video_level/exp{eid}_{task}_{lab}_{nc}class.csv",index=False)
    rows.append(dict(exp_id=eid,name=name,arm=arm,task=task,label_type=lab,n_subs=ns,n_class=nc,n_folds=nf,
        chance=round(100/nc,2),lds="bidirectional",running_norm=rn,per_subject_norm="on",reorder="on",
        overall_acc=round(acc,4),mean_subject_acc=round(float(sub.mean()),4),std_subject_acc=round(float(sub.std()),4),
        ratio_to_chance=round(acc/(100/nc),3),macro_recall=round(float(np.mean(rec)),4),
        n_recall_gt20=int(sum(1 for r in rec if r>20)),
        recall_by_class="|".join(f"{r:.1f}" for r in rec),
        within_video_const_pct=(round(float(const),2) if const==const else ""),
        spearman_rho=(round(rho,4) if rho==rho else ""),spearman_p=(round(p,4) if p==p else ""),
        oof_source=os.path.relpath(f,SRC)))
df=pd.DataFrame(rows); df.to_csv(f"{REPO}/results/metrics.csv",index=False)
print(df[["exp_id","name","n_subs","n_class","chance","overall_acc","ratio_to_chance","n_recall_gt20","spearman_rho"]].to_string(index=False))
print(f"\n{len(df)} 行 → results/metrics.csv")
