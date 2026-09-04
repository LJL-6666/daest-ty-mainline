#!/bin/bash
# 实验 01–07：主线正确版基线（做逐被试归一化 + 做播放序重排 · 双向 LDS）
# 用法: DATA=TY9_movie bash scripts/10_baselines.sh   或不带 DATA 跑全部
set -uo pipefail
cd "$(dirname "$0")/.." ; source scripts/00_check_env.sh
PY="${DAEST_PYTHON:-python}"; OUT="${DAEST_RUNS_ROOT:?见 env/PATHS.md}"
G=(${HEALTHY_GPUS:-0}); RUN=1
ALL=(FACED_05_47 TY9_movie TY9_communication TY8_movie_self TY8_communication_self TY9_movie_matched51 TY9_communication_matched51)
LIST=(${DATA:-${ALL[@]}})
for D in "${LIST[@]}"; do
  [ "$D" = FACED_05_47 ] && V=10 || V=5
  CP="$OUT/${D}_mainline_cp/runs/cp"; WK="$OUT/${D}_mainline_cp/runs/data_work"
  echo "########## $D  ($V 折)"
  # ① 对比预训练（若已有 encoder 可跳过，用 ext_fea.load_cp_dir 指过去）
  [ -z "${SKIP_TRAIN_EXT:-}" ] && for f in $(seq 0 $((V-1))); do
    CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/train_ext.py data=$D log.run=$RUN \
      "log.cp_dir='$CP'" "data.work_dir='$WK'" train.valid_method=$V train.fold_index=$f train.gpus=[0]
  done
  # ② 提特征（双向 LDS）  ③ 训分类头
  for f in $(seq 0 $((V-1))); do
    CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/extract_fea.py data=$D log.run=$RUN \
      "log.cp_dir='$CP'" "data.work_dir='$WK'" ext_fea.mode=me \
      ext_fea.use_lds=true ext_fea.lds_given_all=1 \
      train.valid_method=$V train.fold_index=$f train.gpus=[0] train.num_workers=0
  done
  for f in $(seq 0 $((V-1))); do
    CUDA_VISIBLE_DEVICES=${G[$((f % ${#G[@]}))]} $PY src/train_mlp.py data=$D log.run=$RUN \
      "log.cp_dir='$CP'" "data.work_dir='$WK'" ext_fea.mode=me mlp.wd=0.0022 \
      train.valid_method=$V train.fold_index=$f train.gpus=[0] train.num_workers=0 &
  done; wait
done
