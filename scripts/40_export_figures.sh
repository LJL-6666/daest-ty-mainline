#!/bin/bash
# 导出逐窗 OOF → 生成 results/metrics.csv 与视频级 OOF → 出图
set -uo pipefail
cd "$(dirname "$0")/.." ; PY="${DAEST_PYTHON:-python}"; OUT="${DAEST_RUNS_ROOT:?}"
echo "=== 逐窗 OOF 导出 ==="
for p in faced ty9_movie ty9_communication ty8_movie_self ty8_communication_self \
         ty9_movie_matched51 ty9_communication_matched51; do
  $PY src/export_window_predictions_oof.py --preset "$p" || echo "  ⚠ $p 跳过（路径需按实际覆盖）"
done
echo "=== 汇总为 metrics.csv + 视频级 OOF ==="
$PY scripts/build_metrics.py
echo "=== 逐被试准确率图与混淆矩阵 ==="
$PY src/visualize_ty9_movie.py         --run 1 --mode me || true
$PY src/visualize_ty9_communication.py --run 1 --mode me || true
echo "完成，见 results/"
