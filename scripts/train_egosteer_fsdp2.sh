#!/bin/bash

# Master-only multi-node Accelerate FSDP launcher using pdsh.
# Usage: bash scripts/train_egosteer_fsdp2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# ---------------- CONFIGURATION ----------------
NODES=(
    # One entry per worker node (hostname or IP reachable over SSH).
    "node-01"
    "node-02"
)

SSH_USER=""
GPUS_PER_NODE=8
MASTER_PORT=18276

SCRIPT="train.py"
ARGS="experiment=egosteer_qwen3_vl"
# -----------------------------------------------

if ! command -v pdsh >/dev/null 2>&1; then
    echo "pdsh is not installed on this node."
    exit 1
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

MASTER_ADDR=${NODES[0]}
NNODES=${#NODES[@]}
TOTAL_PROCESSES=$((GPUS_PER_NODE * NNODES))
HOSTLIST=$(IFS=,; echo "${NODES[*]}")

export PDSH_RCMD_TYPE=ssh

ssh_target_prefix() {
    if [ -n "$SSH_USER" ]; then
        printf "%s@" "$SSH_USER"
    fi
}

remote_cleanup() {
    pdsh -S -R exec -w "$HOSTLIST" \
        ssh -o BatchMode=yes "$(ssh_target_prefix)%h" \
        "pkill -f 'torchrun' || true; pkill -f 'train.py' || true" || true
}

cleanup() {
    local exit_code="${1:-130}"
    trap - INT TERM EXIT

    echo
    echo "Stopping remote training processes on all nodes..."
    remote_cleanup
    exit "$exit_code"
}

trap 'cleanup 130' INT TERM
trap 'cleanup $?' EXIT

echo "Launching training on $NNODES nodes from master..."
echo "Project directory: $PROJECT_DIR"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Total Processes: $TOTAL_PROCESSES"
echo "Hostlist: $HOSTLIST"

# echo "Cleaning up previous runs on all nodes..."
# remote_cleanup
# echo "Cleanup complete."

read -r -d '' REMOTE_SCRIPT <<EOF || true
cd "$PROJECT_DIR"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate egosteer
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 # support capture scalar outputs in transformers
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_LOGS="recompiles"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec torchrun \\
    --nnodes="$NNODES" \\
    --node_rank=__NODE_RANK__ \\
    --master_addr="$MASTER_ADDR" \\
    --master_port="$MASTER_PORT" \\
    --nproc_per_node="$GPUS_PER_NODE" \\
    --no-python \\
    bash "$PROJECT_DIR/scripts/numa_bind_wrapper.sh" \\
    "$SCRIPT" \\
    $ARGS
EOF

PDSH_COMMAND=${REMOTE_SCRIPT//__NODE_RANK__/%n}

echo "Starting pdsh launcher. Press Ctrl+C to stop all nodes."
pdsh -S -R exec -w "$HOSTLIST" \
    ssh -tt -o BatchMode=yes "$(ssh_target_prefix)%h" \
    "bash -lc '$PDSH_COMMAND'"

trap - EXIT
echo "Training finished."
