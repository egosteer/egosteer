#!/bin/bash

# Single-node FSDP2 launcher using torchrun.
# Usage: bash scripts/train_egosteer_fsdp2_single_node.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ---------------- CONFIGURATION ----------------
SCRIPT="train.py"
ARGS="experiment=egosteer_qwen3_vl"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-18276}"
GPU_COUNT="8"
# -----------------------------------------------

cd "$PROJECT_DIR"

export NCCL_DEBUG="INFO"
export NCCL_TIMEOUT="3600"
export NCCL_ASYNC_ERROR_HANDLING="1"
export NCCL_IB_DISABLE="0"
export MALLOC_TRIM_THRESHOLD_="0"
export MALLOC_MMAP_THRESHOLD_="65536"
export MALLOC_ARENA_MAX="2"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="1"
export PYTHONUNBUFFERED=1

echo "Launching single-node FSDP2 training..."
echo "Project Directory: $PROJECT_DIR"
echo "GPU Count: $GPU_COUNT"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

exec torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPU_COUNT" \
    --no-python \
    bash "$SCRIPT_DIR/numa_bind_wrapper.sh" \
    "$SCRIPT" \
    $ARGS
