"""LR-scheduler builders used by training workspaces.

Isolated here so multiple workspaces can share the same staged-training
schedule recipe without importing the training workspace class.
"""

from __future__ import annotations

from functools import partial
from typing import Iterable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Source: transformers/src/transformers/optimization.py
# `_get_cosine_schedule_with_warmup_lr_lambda` / `_get_linear_schedule_with_warmup_lr_lambda`
# are the bare lambda factories used internally by get_cosine_schedule_with_warmup /
# get_linear_schedule_with_warmup. We use them directly because we build per-group
# lambdas that share the same decay profile across groups.
from transformers.optimization import (
    _get_cosine_schedule_with_warmup_lr_lambda,
    _get_linear_schedule_with_warmup_lr_lambda,
)


_SCHEDULE_FN = {
    "cosine": _get_cosine_schedule_with_warmup_lr_lambda,
    "linear": _get_linear_schedule_with_warmup_lr_lambda,
}

def _zero_lr(step: int) -> float:
    del step
    return 0.0


def _vlm_freeze_lr(step: int, *, vlm_freeze_steps: int, vlm_base_lambda) -> float:
    if step < vlm_freeze_steps:
        return 0.0
    return vlm_base_lambda(step - vlm_freeze_steps)


def build_lr_scheduler(
    optimizer: Optimizer,
    schedule_name: str,
    num_warmup_steps: int,
    num_training_steps: int,
    vlm_group_indices: Iterable[int],
    vlm_freeze_steps: int = 0,
    vlm_rewarmup_steps: int = 0,
) -> LambdaLR:
    """Build a per-group LambdaLR scheduler.

    Non-VLM groups follow the standard warmup -> cosine/linear decay.
    VLM groups can stay at lr=0 for ``vlm_freeze_steps``, then optionally
    run their own warmup over ``vlm_rewarmup_steps``, and finally follow
    the same cosine/linear decay for the remaining step budget.

    Args:
        optimizer: Optimizer whose param_groups we schedule.
        schedule_name: "cosine" or "linear" (controls the decay profile).
        num_warmup_steps: Warmup steps for non-VLM groups.
        num_training_steps: Total training update steps.
        vlm_group_indices: Indices of VLM groups inside ``optimizer.param_groups``.
        vlm_freeze_steps: Number of steps VLM groups stay at lr=0.
        vlm_rewarmup_steps: Warmup length applied to VLM groups after the freeze.
    """
    if schedule_name not in _SCHEDULE_FN:
        raise ValueError(f"Unsupported lr_scheduler: {schedule_name}")
    lr_lambda_fn = _SCHEDULE_FN[schedule_name]
    # cosine variant requires num_cycles; linear ignores it via **kwargs.
    extra_kwargs = {"num_cycles": 0.5} if schedule_name == "cosine" else {}

    base_lambda = partial(
        lr_lambda_fn,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **extra_kwargs,
    )

    vlm_group_indices = set(vlm_group_indices)

    def make_lambda(group_idx: int):
        if group_idx not in vlm_group_indices:
            return base_lambda

        if vlm_freeze_steps <= 0 and vlm_rewarmup_steps <= 0:
            return base_lambda

        if vlm_freeze_steps >= num_training_steps:
            return partial(_zero_lr)

        vlm_total = max(1, num_training_steps - vlm_freeze_steps)
        vlm_warmup = min(max(0, vlm_rewarmup_steps), vlm_total)
        vlm_base_lambda = partial(
            lr_lambda_fn,
            num_warmup_steps=vlm_warmup,
            num_training_steps=vlm_total,
            **extra_kwargs,
        )
        return partial(
            _vlm_freeze_lr,
            vlm_freeze_steps=vlm_freeze_steps,
            vlm_base_lambda=vlm_base_lambda,
        )

    num_groups = len(optimizer.param_groups)
    lr_lambdas = [make_lambda(i) for i in range(num_groups)]
    return LambdaLR(optimizer, lr_lambdas)
