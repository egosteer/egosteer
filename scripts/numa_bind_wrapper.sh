#!/bin/bash
# Per-rank NUMA binding wrapper.
#
# Invoked by torchrun with --no-python:
#   torchrun ... --no-python bash scripts/numa_bind_wrapper.sh train.py [args...]
#
# Reads LOCAL_RANK from env (torchrun sets it per child), discovers the NUMA
# node physically attached to that GPU via sysfs, then exec's Python under
# numactl so both CPU scheduling and memory allocation stay NUMA-local.
# Dataloader workers fork from this process and inherit the binding.
#
# Reference: https://man7.org/linux/man-pages/man8/numactl.8.html
# Reference: https://docs.nvidia.com/deploy/mps/index.html#topic_5_5

set -e

LOCAL_RANK="${LOCAL_RANK:-0}"

# Autodetect NUMA from sysfs via GPU PCI BDF.
# nvidia-smi outputs BDF as "00000000:17:00.0" (8-char domain);
# sysfs uses "0000:17:00.0" (4-char domain), so strip the first 4 chars.
GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$LOCAL_RANK" \
    | tr 'A-Z' 'a-z' | cut -c5-)
NUMA_NODE=$(cat "/sys/bus/pci/devices/${GPU_PCI}/numa_node" 2>/dev/null || echo -1)

# Fallback for kernels/containers where sysfs numa_node is -1 (rare).
# Standard 8-GPU 2-socket layout: GPU 0-3 -> NUMA 0, GPU 4-7 -> NUMA 1.
if [ "$NUMA_NODE" -lt 0 ]; then
    NUMA_NODE=$((LOCAL_RANK / 4))
fi

echo "[rank $LOCAL_RANK] bound to NUMA $NUMA_NODE (GPU pci $GPU_PCI)" >&2

exec numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
    python -u "$@"
