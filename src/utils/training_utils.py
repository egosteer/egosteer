import gc
import math
import os
import random
import sys
import threading
import pty
import re
import time
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist


class TrainingState:
    """A simple class to encapsulate all scalar training states that need to be saved."""
    def __init__(self, epoch: int = 0, update_step: int = 0, global_step: int = 0):
        self.epoch = epoch
        self.update_step = update_step
        self.global_step = global_step

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "update_step": self.update_step,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.update_step = state_dict["update_step"]
        self.global_step = state_dict["global_step"]


class DeviceTransferWrapper:
    """Wraps a dataloader to transfer batches to the target device on iteration."""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.batch_size = getattr(dataloader, "batch_size", 1)

    def __iter__(self):
        for batch in self.dataloader:
            yield {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }


@contextmanager
def tee_output_to_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_file = open(path, "ab", buffering=0)

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    read_fd = None
    try:
        master_fd, slave_fd = pty.openpty()
        os.dup2(slave_fd, stdout_fd)
        os.dup2(slave_fd, stderr_fd)
        os.close(slave_fd)
        read_fd = master_fd
    except Exception:
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, stdout_fd)
        os.dup2(write_fd, stderr_fd)
        os.close(write_fd)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass

    ansi_escape = re.compile(rb"\x1B\[[0-?]*[ -/]*[@-~]")

    def _reader():
        while True:
            try:
                data = os.read(read_fd, 4096)
            except OSError:
                break
            if not data:
                break
            os.write(saved_stdout_fd, data)
            log_file.write(ansi_escape.sub(b"", data))
            log_file.flush()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        try:
            os.close(read_fd)
        except Exception:
            pass
        reader_thread.join(timeout=1)
        log_file.flush()
        log_file.close()


def capture_output_to_training_log(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        log_path = os.path.join(self.output_dir, "training.log")
        with tee_output_to_file(log_path):
            return func(self, *args, **kwargs)
    return wrapper


def scalar_metric_value(value):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        # FSDP2 DTensor with _NormPartial placement: .item() only returns the
        # local shard's value without triggering all-reduce, giving
        # full_norm / sqrt(world_size) instead of the true global norm.
        # Calling .full_tensor() forces the reduction (x^p -> allreduce_sum -> x^(1/p)).
        # See: https://github.com/pytorch/pytorch/issues/144054
        #      https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py
        if hasattr(value, 'full_tensor'):
            value = value.full_tensor()
        return value.detach().float().cpu().item()
    return value


def params_l2_norm(params):
    params = [p for p in params if p is not None]
    if not params:
        return 0.0
    norm = torch.nn.utils.get_total_norm(params, norm_type=2.0)
    return scalar_metric_value(norm)


# Adapted from TorchTitan + TorchTNT to avoid GC stragglers in distributed
# training.  All ranks disable automatic GC and collect at the same
# deterministic step, so no single rank stalls at the next NCCL collective.
#
# Two-tier collection (following TorchTNT's pattern):
#   - gen-1 every ``gc_freq`` steps   — lightweight, catches short-lived cycles
#   - gen-2 every ``full_gc_freq`` steps — full sweep, prevents long-lived
#     reference-cycle memory buildup that gen-1 alone cannot reclaim
#
# Sources:
#   https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
#   https://github.com/pytorch/tnt  (torchtnt.framework.callbacks.GarbageCollector)
class GarbageCollection:
    def __init__(
        self,
        gc_freq: int = 100,
        full_gc_freq: int = 1000,
        debug: bool = False,
    ):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        assert full_gc_freq >= gc_freq, "full_gc_freq should be >= gc_freq"
        self.gc_freq = gc_freq
        self.full_gc_freq = full_gc_freq
        self.debug = debug
        gc.disable()
        self.collect("Initial GC collection", generation=2)
        if debug:
            from torch.utils.viz._cycles import warn_tensor_cycles
            if torch.distributed.get_rank() == 0:
                warn_tensor_cycles()

    def run(self, step_count: int):
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            gc.collect()
            return

        if step_count < 2:
            return
        if step_count % self.full_gc_freq == 0:
            self.collect("Performing full (gen-2) GC collection", generation=2)
        elif step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")

    def finalize(self):
        gc.enable()

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        elapsed = time.monotonic() - begin
        if elapsed > 0.05:
            print(f"[GC] {reason} took {elapsed:.2f}s")


def data_worker_init(worker_id: int) -> None:
    """DataLoader worker_init_fn.

    Re-enables Python GC (main process disables it via GarbageCollection
    helper) and configures the data-quality skip logger handler. The logger
    setup is a no-op on fork once the parent has run it; needed under spawn
    where workers re-import the module fresh.
    """
    del worker_id
    gc.enable()
    # Re-apply file_system sharing strategy in case workers were spawned
    # (fresh interpreter) instead of forked from the rank process.
    import torch.multiprocessing as _torch_mp
    _torch_mp.set_sharing_strategy("file_system")
    from src.dataset.sanity_checks import configure_logger
    configure_logger()


class FullMemoryTracker:
    def __init__(self, model):
        self.model = model
        self.stats = {}
        self.hooks = []

    def _get_tensor_mem(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.element_size() * tensor.nelement() / (1024**2)
        return 0

    def hook_before(self, module, input, name):
        # Record the GPU memory state before entering this layer
        torch.cuda.synchronize()  # Force synchronization for accurate measurement, though it slows things down
        module._mem_before = torch.cuda.memory_allocated()

    def hook_after(self, module, input, output, name):
        # Record the GPU memory state after leaving this layer
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        # Compute the GPU memory increase during this layer's execution (this includes activations and intermediate variables)
        diff = (mem_after - module._mem_before) / (1024**2)

        # Compute the size of parameters and gradients
        param_mem = sum(p.element_size() * p.nelement() for p in module.parameters(recurse=False)) / (1024**2)
        grad_mem = sum(p.grad.element_size() * p.grad.nelement() if p.grad is not None else 0
                       for p in module.parameters(recurse=False)) / (1024**2)

        if name not in self.stats:
            self.stats[name] = {'param': param_mem, 'peak_delta': 0, 'grad': 0, 'output': 0}

        self.stats[name]['peak_delta'] = max(self.stats[name]['peak_delta'], diff)
        self.stats[name]['grad'] = max(self.stats[name]['grad'], grad_mem)
        self.stats[name]['output'] = max(self.stats[name]['output'], self._get_tensor_mem(output))

    def track(self):
        for name, module in self.model.named_modules():
            # Filter out layers that are too deeply nested; only the main Blocks are of interest
            if len(list(module.children())) <= 3:
                h_pre = module.register_forward_pre_hook(lambda m, i, n=name: self.hook_before(m, i, n))
                h_post = module.register_forward_hook(lambda m, i, o, n=name: self.hook_after(m, i, o, n))
                self.hooks.extend([h_pre, h_post])

    def report(self):
        print(f"\n{'Module Name':<50} | {'Param(MB)':<10} | {'Grad(MB)':<10} | {'Output(MB)':<12} | {'Net-Delta(MB)':<12}")
        print("-" * 105)
        # Sort by delta to identify the real memory hogs
        sorted_items = sorted(self.stats.items(), key=lambda x: x[1]['peak_delta'], reverse=True)
        for name, s in sorted_items[:200]:
            print(f"{name[:50]:<50} | {s['param']:>10.1f} | {s['grad']:>10.1f} | {s['output']:>12.1f} | {s['peak_delta']:>12.1f}")
        print(f"Total memory usage: {sum(s['peak_delta'] for s in self.stats.values()):.1f} MB")

    def stop(self):
        for h in self.hooks:
            h.remove()


def reset_run_seed(base_seed: int, dynamic_data_seed: bool, rank: int) -> int:
    """Reset torch/numpy/random seeds for the current rank.

    When ``dynamic_data_seed`` is True, rank 0 samples an epoch-level seed
    from wall-clock time and broadcasts it so all ranks agree. The final
    per-rank seed is ``base_seed (+ timestamp) + rank`` to decorrelate any
    per-rank augmentation without losing reproducibility of the global
    schedule.
    """
    timestamp_seed = None
    run_seed = base_seed
    if dynamic_data_seed:
        objects = [int(time.time())] if rank == 0 else [None]
        dist.broadcast_object_list(objects, src=0)
        timestamp_seed = int(objects[0])
        run_seed = base_seed + timestamp_seed

    per_device_seed = run_seed + rank
    torch.manual_seed(per_device_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(per_device_seed)
    np.random.seed(per_device_seed % (2**32 - 1))
    random.seed(per_device_seed)

    dist.barrier()
    if rank == 0:
        if dynamic_data_seed:
            print(f"Using runtime seed: {run_seed} (base={base_seed}, timestamp={timestamp_seed})")
        else:
            print(f"Using fixed seed: {run_seed}")
    return run_seed


def build_param_groups(
    model,
    optimizer_cfg,
    *,
    train_vlm: bool,
) -> tuple[list[dict], set[int], list[torch.nn.Parameter]]:
    """Collect AdamW parameter groups for EgoSteer's components.

    Components produce two param groups each (decay / no-decay by tensor
    rank). Returns:
        groups:             list of dicts suitable for ``torch.optim.AdamW``.
        vlm_group_indices:  indices of VLM groups inside ``groups`` — used
                            by the lr scheduler to apply a separate
                            freeze/rewarmup curve.
        vlm_param_refs:     flat list of VLM Parameter objects, used by
                            the train loop to drop ``.grad`` during the
                            staged-freeze window so AdamW skips updates.

    Must be called AFTER ``fully_shard()`` so that the references point at
    the post-shard DTensor-wrapped Parameters (pre-shard refs get orphaned
    because FSDP2 replaces ``module._parameters[name]``).
    """
    groups: list[dict] = []

    groups.extend(grouped_parameters(model.action_expert_parameters, optimizer_cfg.action))

    vlm_group_indices: set[int] = set()
    vlm_param_refs: list[torch.nn.Parameter] = []
    if train_vlm:
        vlm_groups = grouped_parameters(model.trainable_vlm_parameters, optimizer_cfg.vlm)
        start = len(groups)
        groups.extend(vlm_groups)
        vlm_group_indices = set(range(start, start + len(vlm_groups)))
        for group in vlm_groups:
            vlm_param_refs.extend(group["params"])

    groups.extend(grouped_parameters(model.ar_action_heads_parameters, optimizer_cfg.ar_action_heads))

    if getattr(model, "use_world_model", False):
        groups.extend(grouped_parameters(model.world_model_parameters, optimizer_cfg.world_model))

    all_params = [p for group in groups for p in group["params"]]
    trainable_ids = {id(p) for p in all_params}
    for idx, param in enumerate(all_params):
        assert param.requires_grad, (
            f"Parameter at index {idx} is in optimizer groups but requires_grad is False"
        )
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert id(param) in trainable_ids, (
                f"Parameter '{name}' requires grad but is NOT in the optimizer parameters list"
            )

    return groups, vlm_group_indices, vlm_param_refs


def grouped_parameters(param_list, component_cfg) -> list[dict]:
    """Split ``param_list`` into AdamW decay / no-decay groups.

    Convention: tensors with ``ndim >= 2`` decay; biases and norm params
    (ndim < 2) use weight_decay=0.
    """
    params = [p for p in param_list if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    # Plain list so FSDP2 checkpoint serialization (torch._iterate_state_dict)
    # accepts it.
    betas = list(component_cfg.betas)
    return [
        {"params": decay_params, "weight_decay": component_cfg.weight_decay, "lr": component_cfg.lr, "betas": betas},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": component_cfg.lr, "betas": betas},
    ]


def clip_and_check_grads(
    model,
    cfg_clipping,
    *,
    train_vlm: bool,
    vlm_freeze_active: bool,
    rank: int,
    update_step: int,
    global_step: int,
) -> tuple[dict[str, torch.Tensor] | None, bool]:
    """Clip per-component grads and detect non-finite norms.

    Returns ``(norms, step_skipped)``:
        norms:        {component_name: pre-clip L2 norm} or None when
                      clipping is disabled.
        step_skipped: True when any component's norm is NaN/Inf — the
                      caller should skip optimizer.step() for this micro-batch.

    clip_grad_norm_ returns a DTensor-reduced scalar under FSDP2, so every
    rank sees the same value and decides identically without extra
    collectives.
    """
    if not cfg_clipping.enabled:
        return None, False

    max_norm = cfg_clipping.max_grad_norm
    part_params = {
        "action_expert": model.action_expert_parameters,
        "ar_action_heads": model.ar_action_heads_parameters,
    }
    if getattr(model, "use_world_model", False):
        part_params["world_model"] = model.world_model_parameters
    if train_vlm and not vlm_freeze_active:
        part_params["vision"] = model.trainable_vision_parameters
        part_params["text"] = model.trainable_text_parameters

    norms = {
        name: torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=True)
        for name, params in part_params.items()
    }

    if any(not math.isfinite(scalar_metric_value(n)) for n in norms.values()):
        if rank == 0:
            readable = {k: scalar_metric_value(v) for k, v in norms.items()}
            print(
                f"[WARN] Non-finite grad norm at update_step={update_step} "
                f"global_step={global_step}: {readable}. Skipping step."
            )
        return norms, True

    return norms, False


def build_training_step_log(
    workspace,
    *,
    include_full_metrics: bool,
    raw_loss: dict[str, torch.Tensor] | None = None,
    part_grad_norms: dict[str, torch.Tensor] | None = None,
    batch: dict | None = None,
    step_perf_start: float | None = None,
    data_wait_sec: float | None = None,
    training_start_time: float | None = None,
    total_samples_processed: int = 0,
    train_vlm: bool = False,
) -> dict:
    """Build the per-step wandb log dict.

    The base fields (ids, lr, vlm state) are always included. Full metrics
    (timings, grad norms, weight norms, raw loss components) are added
    only when ``include_full_metrics=True`` — typically on logging steps.
    """
    current_group_lrs, non_vlm_group_indices = workspace.get_param_group_lrs()
    current_lr = (
        current_group_lrs[non_vlm_group_indices[0]]
        if non_vlm_group_indices else current_group_lrs[0]
    )

    step_log: dict = {
        "global_step": workspace.global_step,
        "update_step": workspace.update_step,
        "epoch": workspace.epoch,
        "lr/non_vlm": current_lr,
        "vlm_freeze_active": float(workspace.is_vlm_freeze_active()),
    }
    if workspace.vlm_group_indices:
        step_log["lr/vlm"] = current_group_lrs[min(workspace.vlm_group_indices)]

    if not include_full_metrics:
        return step_log

    step_wall_time = time.time()
    step_time_sec = time.perf_counter() - step_perf_start
    batch_size_local = batch["input_ids"].shape[0]
    elapsed_time_sec = step_wall_time - training_start_time
    step_log.update({
        "time/elapsed_sec": elapsed_time_sec,
        "time/step_sec": step_time_sec,
        "time/data_wait_sec": data_wait_sec,
        "time/avg_samples_per_sec": total_samples_processed / elapsed_time_sec if elapsed_time_sec > 0 else 0,
        "time/samples_per_sec": batch_size_local / step_time_sec if step_time_sec > 0 else 0,
    })

    if part_grad_norms is not None:
        for component in ("action_expert", "ar_action_heads", "vision", "text", "world_model"):
            if component in part_grad_norms:
                step_log[f"grad_norm/{component}"] = part_grad_norms[component]

    model = workspace.model
    with torch.no_grad():
        if train_vlm:
            step_log["weight_norm/vision"] = params_l2_norm(model.trainable_vision_parameters)
            step_log["weight_norm/text"] = params_l2_norm(model.trainable_text_parameters)
        step_log["weight_norm/action"] = params_l2_norm(model.action_expert_parameters)
        step_log["weight_norm/ar_action_heads"] = params_l2_norm(model.ar_action_heads_parameters)
        if getattr(model, "use_world_model", False):
            step_log["weight_norm/world_model"] = params_l2_norm(model.world_model_parameters)

    if raw_loss is not None:
        for key, value in raw_loss.items():
            step_log[f"train_loss/{key}"] = value.item()

    return step_log
