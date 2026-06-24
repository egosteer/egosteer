#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# ---------------- CONFIGURATION ----------------
SCRIPT="train.py"
ARGS="experiment=egosteer_qwen3_vl"
# -----------------------------------------------

# Container-cloud topology. Unlike the pdsh launcher, this script is started
# once inside EACH node's container; the platform injects the topology as
# environment variables, which torchrun reads below to wire up the worker group.
#
# Replace the XXX_* names on the right-hand side with the variables your platform
# actually exports:
#   MASTER_ADDR     host/IP of node 0 (rendezvous endpoint)
#   MASTER_PORT     rendezvous port on node 0
#   MACHINE_RANK    this node's index in [0, NNODES)
#   NNODES          total number of nodes
#   GPUS_PER_NODE   GPUs visible to each container
# RDMA_IFNAME is the high-speed NIC used for NCCL/Gloo; override if not eth0.
MASTER_ADDR="$XXX_WORKER_0_HOST"
MASTER_PORT="$XXX_WORKER_0_PORT"
MACHINE_RANK="$XXX_ROLE_INDEX"
NNODES="$XXX_WORKER_NUM"
GPUS_PER_NODE="$XXX_WORKER_GPU"
RDMA_IFNAME="${RDMA_IFNAME:-eth0}"

cd "$PROJECT_DIR"

export NCCL_SOCKET_FAMILY="AF_INET"
export GLOO_SOCKET_IFNAME="$RDMA_IFNAME"
export TP_SOCKET_IFNAME="$RDMA_IFNAME"
export NCCL_SOCKET_IFNAME="$RDMA_IFNAME"
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

echo "Launching Container Cloud FSDP2 training..."
echo "Project Directory: $PROJECT_DIR"
echo "Socket Interface: $RDMA_IFNAME"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Machine Rank: $MACHINE_RANK"
echo "Num Machines: $NNODES"
echo "GPUs Per Node: $GPUS_PER_NODE"

exec torchrun \
    --nnodes="$NNODES" \
    --node_rank="$MACHINE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --no-python \
    bash "$SCRIPT_DIR/numa_bind_wrapper.sh" \
    "$SCRIPT" \
    $ARGS