from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

log = logging.getLogger(__name__)


def _collect_fsdp_modules(module: torch.nn.Module) -> list:
    """Collect all FSDPModule instances in the module tree (FSDP2)."""
    # Import here to avoid a hard dependency when FSDP is not used.
    from torch.distributed.fsdp import FSDPModule
    return [m for m in module.modules() if isinstance(m, FSDPModule)]


@dataclass
class DistributedContext:
    """Holds process group topology and device assignment."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    mesh: DeviceMesh | None  # None for single-node plain FSDP


def init_distributed(
    backend: str = "nccl"
) -> DistributedContext:
    """Initialize the distributed process group and optionally create an HSDP mesh.

    When running on multiple nodes (detected via LOCAL_WORLD_SIZE from
    torchrun), a 2D DeviceMesh is created automatically:
      - dim 0 ("replicate"): across nodes — gradient all-reduce only
      - dim 1 ("shard"):     within a node — all-gather / reduce-scatter via NVLink

    On a single node the mesh is left as None for plain 1D FSDP sharding.

    Environment variables read:
        - LOCAL_RANK, LOCAL_WORLD_SIZE
            Reference: torch/distributed/elastic/agent/server/local_elastic_agent.py:305-326
        - NCCL_TIMEOUT (seconds, default 3600)        

    HSDP mesh dim convention (dim0=replicate, dim1=shard):
    Reference: torch/distributed/fsdp/_fully_shard/_fsdp_init.py:60-69

    Returns:
        DistributedContext with rank, device, and optional HSDP mesh.
    """
    nccl_timeout = int(os.environ.get("NCCL_TIMEOUT", 3600))
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=nccl_timeout))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Auto-detect multi-node topology for HSDP
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    num_nodes = world_size // gpus_per_node

    mesh = None
    if num_nodes > 1:
        # 2D mesh: dim 0 = replicate (across nodes), dim 1 = shard (within node)
        # Reference: torch/distributed/device_mesh.py:1460-1469
        mesh = init_device_mesh(
            "cuda",
            mesh_shape=(num_nodes, gpus_per_node),
            mesh_dim_names=("replicate", "shard"),
        )

    if rank == 0:
        if mesh is not None:
            log.info(
                "Distributed init: backend=%s, world_size=%d, "
                "HSDP mesh=%d nodes x %d GPUs/node (shard within node via NVLink)",
                backend, world_size, num_nodes, gpus_per_node,
            )
        else:
            log.info(
                "Distributed init: backend=%s, world_size=%d, plain FSDP (single node)",
                backend, world_size,
            )

        # Report CPU affinity so we can tell whether scripts/numa_bind_wrapper.sh
        # took effect. Expected on a 2-socket box with wrapper: ~half of total
        # cores. Full core count means wrapper was bypassed.
        import psutil
        affinity = sorted(psutil.Process().cpu_affinity())
        log.info(
            "CPU affinity on rank 0: %d cores, range [%d-%d]",
            len(affinity), affinity[0], affinity[-1],
        )

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        mesh=mesh,
    )


def build_mixed_precision_policy(use_bf16: bool) -> MixedPrecisionPolicy | None:
    """Build the project's standard mixed-precision policy.

    The recipe: fp32 master params with bf16 compute + bf16 output.
    reduce_dtype stays fp32 for numerically safe gradient all-reduce.
    cast_forward_inputs=True lets FSDP recursively cast floating-point
    forward inputs to param_dtype; non-floating tensors (int/bool/uint8)
    pass through untouched, so caller-side dtype casting is unnecessary.

    Returns None when use_bf16 is False (full fp32 training).

    Reference: torch/distributed/fsdp/_fully_shard/_fsdp_api.py:14-53
    Reference: torch/distributed/fsdp/_fully_shard/_fsdp_common.py:171-178
        (_cast_fp_tensor: skips non-floating tensors)
    """
    if not use_bf16:
        return None
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )


def apply_fsdp2(
    model: torch.nn.Module,
    wrap_classes: tuple[type, ...],
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool = False,
    mp_policy: MixedPrecisionPolicy | None = None,
    enable_prefetch: bool = True,
) -> None:
    """Apply FSDP2 sharding to a model by wrapping matching sub-modules, then root.

    Sub-modules whose type matches any entry in wrap_classes are sharded
    first (leaf-to-root order from module iteration). The root module is
    always sharded last.

    With a 2D mesh this becomes HSDP: parameters are sharded on dim 1
    and replicated on dim 0. With mesh=None, this is plain 1D FSDP.

    Reference: torch/distributed/fsdp/_fully_shard/_fully_shard.py:90-99

    Args:
        model: The model to shard. Modified in place.
        wrap_classes: Tuple of module types to individually shard before root.
        mesh: 2D DeviceMesh for HSDP, or None for plain FSDP.
        reshard_after_forward: Whether to reshard parameters after each
            forward pass. False keeps params unsharded between forward and
            backward, trading memory for speed.
        mp_policy: Mixed precision policy. When None, all computation
            stays in the model's current dtype.
        enable_prefetch: If True, build explicit forward/backward prefetch
            chains between consecutive sharded modules of the same class.
            This lets FSDP overlap all-gather / reduce-scatter / (HSDP)
            all-reduce with the compute of the previous block, instead of
            leaving the collectives exposed at the end of each block.
            Chains are scoped per parent ModuleList so we never prefetch
            across expert boundaries (e.g. vision -> text) where forward
            execution order is not guaranteed to match module iteration.
    """
    fsdp_kwargs: dict = {"reshard_after_forward": reshard_after_forward}
    if mesh is not None:
        fsdp_kwargs["mesh"] = mesh
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy

    # (parent_id, class) -> list of sharded child modules in registration order.
    # Siblings under the same ModuleList / Sequential execute in index order,
    # so they are safe to chain for prefetch. Modules under different parents
    # or of different classes stay in separate chains.
    prefetch_groups: dict[tuple[int, type], list[torch.nn.Module]] = {}

    # Once a module is wrapped as one FSDP unit, do NOT recurse into its
    # children to wrap them again — that would split the params back out
    # into a nested unit.
    wrapped_ids: set[int] = set()

    for parent in model.modules():
        if id(parent) in wrapped_ids:
            continue
        for child in parent.children():
            if not isinstance(child, wrap_classes):
                continue
            fully_shard(child, **fsdp_kwargs)
            wrapped_ids.add(id(child))
            if enable_prefetch:
                key = (id(parent), type(child))
                prefetch_groups.setdefault(key, []).append(child)

    fully_shard(model, **fsdp_kwargs)

    if not enable_prefetch:
        return

    # Build bidirectional prefetch chain within each group.
    # For a sequence [m0, m1, m2, ...]:
    #   - m_{i}.set_modules_to_forward_prefetch([m_{i+1}])
    #     overlaps m_{i+1}'s all-gather with m_i's forward compute.
    #   - m_{i+1}.set_modules_to_backward_prefetch([m_i])
    #     overlaps m_i's reduce-scatter / HSDP all-reduce with m_{i+1}'s
    #     backward compute, which is the main win under HSDP.
    for chain in prefetch_groups.values():
        for i in range(len(chain) - 1):
            cur, nxt = chain[i], chain[i + 1]
            if hasattr(cur, "set_modules_to_forward_prefetch"):
                cur.set_modules_to_forward_prefetch([nxt])
            if hasattr(nxt, "set_modules_to_backward_prefetch"):
                nxt.set_modules_to_backward_prefetch([cur])
