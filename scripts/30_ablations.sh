#!/bin/bash
# 消融 30–32（讲述臂）：去中性 8 类 / 关 running_norm
# 30: 8类去中性（复用 03 的特征，只换标签与输出头，严格单因素）
# 31/32: 关 running_norm 后的 9 类与 8 类
set -uo pipefail
cd "$(dirname "$0")/.." ; source scripts/00_check_env.sh
PY="${DAEST_PYTHON:-python}"; OUT="${DAEST_RUNS_ROOT:?}"; G=(${HEALTHY_GPUS:-0}); V=5
SRC_CP="$OUT/TY9_communication_mainline_cp/runs/cp"

echo "########## 30  讲述素材8（去中性）：复用 03 特征，仅换标签"
BASE_FEA="$OUT/TY9_communication_mainline_cp/runs/TY9_communication/ext_fea/fea_r1"
N8="$OUT/TY9_communication_8class_cp"; F8="$N8/runs/TY9_communication_8class/ext_fea/fea_r1"
mkdir -p "$F8" "$N8/runs/cp"
for f in 0 1 2 3 4; do ln -sfn "$BASE_FEA/_r1_f${f}_fea_me.npy" "$F8/_r1_f${f}_fea_me.npy"; done
$PY - "$BASE_FEA" "$F8" <<'PYEOF'
import sys,numpy as np
o=np.load(sys.argv[1]+"/onesub_label2.npy"); n=o.copy(); n[o==4]=-1
for k in (5,6,7,8): n[o==k]=k-1
assert sorted(set(n.tolist()))==[-1,0,1,2,3,4,5,6,7] and (n==-1).sum()==120, "标签重映射异常"
np.save(sys.argv[2]+"/onesub_label2.npy", n); print("  标签 ok:", np.bincount(n[n>=0]).tolist(), " 无效", int((n==-1).sum()))
PYEOF
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/train_mlp.py data=TY9_communication_8class_noNeutral log.run=1 \
    "log.cp_dir='$N8/runs/cp'" "data.work_dir='$N8/runs/data_work'" \
    ext_fea.mode=me mlp.wd=0.0022 train.valid_method=$V train.fold_index=$f train.gpus=[0] train.num_workers=0 &
done; wait

echo "########## 31/32  关 running_norm（保留逐被试归一化与双向 LDS）"
NR="$OUT/TY9_communication_noRN_cp"; RUN=904
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/extract_fea.py data=TY9_communication log.run=$RUN \
    "log.cp_dir='$NR/runs/cp'" "data.work_dir='$NR/runs/data_work'" \
    "+ext_fea.load_cp_dir='$SRC_CP'" ext_fea.ckpt_run=1 ext_fea.mode=me \
    ext_fea.use_running_norm=false ext_fea.use_lds=true ext_fea.lds_given_all=1 \
    train.valid_method=$V train.fold_index=$f train.gpus=[0] train.num_workers=0
done
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/train_mlp.py data=TY9_communication log.run=$RUN \
    "log.cp_dir='$NR/runs/cp'" "data.work_dir='$NR/runs/data_work'" \
    ext_fea.mode=me mlp.wd=0.0022 train.valid_method=$V train.fold_index=$f train.gpus=[0] train.num_workers=0 &
done; wait
