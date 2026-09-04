#!/bin/bash
# 实验 08–13：迁移。零样本 = encoder+分类头都借；MLP-FT = 冻结 encoder、仅在目标域重训分类头。
# 两者的 TY 特征都要重新提（用源域 encoder + 目标域自己的 normTrain/running_norm/LDS）。
set -uo pipefail
cd "$(dirname "$0")/.." ; source scripts/00_check_env.sh
PY="${DAEST_PYTHON:-python}"; OUT="${DAEST_RUNS_ROOT:?}"; G=(${HEALTHY_GPUS:-0})
# 源 encoder → 目标数据集
declare -A SRC=( [TY9_movie]=FACED_05_47 [TY9_communication]=FACED_05_47 \
                 [TY9_communication_matched51]=TY9_movie_matched51 )
for TGT in "${!SRC[@]}"; do
  S=${SRC[$TGT]}; SCP="$OUT/${S}_mainline_cp/runs/cp"
  D="$OUT/${S}_to_${TGT}_cp"; V=5
  echo "########## $S → $TGT"
  for f in $(seq 0 $((V-1))); do
    CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/extract_fea.py data=$TGT log.run=1 \
      "log.cp_dir='$D/runs/cp'" "data.work_dir='$D/runs/data_work'" \
      "+ext_fea.load_cp_dir='$SCP'" "+ext_fea.ckpt_dataset=$S" ext_fea.ckpt_run=1 \
      ext_fea.mode=me ext_fea.use_lds=true ext_fea.lds_given_all=1 \
      train.valid_method=$V train.fold_index=$f train.gpus=[0] train.num_workers=0
  done
  # MLP-FT：在目标域重训分类头
  for f in $(seq 0 $((V-1))); do
    CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/train_mlp.py data=$TGT log.run=1 \
      "log.cp_dir='$D/runs/cp'" "data.work_dir='$D/runs/data_work'" \
      ext_fea.mode=me mlp.wd=0.0022 train.valid_method=$V train.fold_index=$f train.gpus=[0] train.num_workers=0 &
  done; wait
  # 零样本：直接用源域 MLP（10 折 softmax 平均），见 src/export_window_predictions_oof.py 与
  # 原仓库 eval_faced_zeroshot_*.py；本仓库以已导出的 OOF 为准。
done
