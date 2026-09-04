#!/bin/bash
# 环境与 GPU 健康检查。source 本脚本后可用 $HEALTHY_GPUS。
# 起因：2026-09-01 一块故障卡把整个 NVIDIA 驱动锁死（uvm_gpu_replayable_faults_isr_lock），
# 触碰它的进程全部进入 D 状态、kill -9 无效，只能重启。故每次跑前必须逐块验、跳过坏卡。
set -uo pipefail
PY="${DAEST_PYTHON:-python}"
echo "=== 环境 ==="
$PY -c "import torch,pytorch_lightning as pl,numpy,pandas,scipy;print(f'  torch {torch.__version__} / lightning {pl.__version__} / numpy {numpy.__version__}')" || exit 1
for v in DAEST_RUNS_ROOT DAEST_PREP_ROOT DAEST_DATA_ROOT; do
  printf "  %-18s %s\n" "$v" "${!v:-❌ 未设置（见 env/PATHS.md）}"; done

echo "=== 逐块验 GPU（各限时 40s，超时即判定为故障）==="
N=$(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$N" -eq 0 ] && { echo "  ⚠ nvidia-smi 无响应，驱动可能已锁死；改用 CPU（train.use_cpu=true）"; export HEALTHY_GPUS=""; return 0 2>/dev/null || exit 0; }
HEALTHY=()
for g in $(seq 0 $((N-1))); do
  if CUDA_VISIBLE_DEVICES=$g timeout 40 $PY -c "
import torch; x=torch.randn(3000,3000,device='cuda')
for _ in range(5): x=x@x/1000
torch.cuda.synchronize()" >/dev/null 2>&1; then
    echo "  GPU$g ✅"; HEALTHY+=("$g")
  else
    echo "  GPU$g ❌ 跳过"
  fi
done
export HEALTHY_GPUS="${HEALTHY[*]}"
echo "可用: ${HEALTHY_GPUS:-（无）}"
D=$(ps -eo stat --no-headers | grep -c '^D')
[ "$D" -gt 0 ] && echo "  ⚠ 检测到 $D 个 D 状态进程，驱动可能已被占死，建议重启"
