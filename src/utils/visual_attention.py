"""Torch-native utilities for analyzing visual-token attention.

Provides reusable helpers for:
  - Stacking sparse per-step/per-layer attention weights into dense tensors
  - Reshaping flat visual tokens into (T_g, H, W) spatial grids
  - Aggregating attention across ODE steps / layers / heads / queries
  - Distribution metrics: spatial entropy, visual-subset renormalization
  - Rendering attention heatmaps as overlays on RGB frames

All functions are torch-first: pass torch.Tensor or np.ndarray (zero-copy via
torch.as_tensor on CPU). Returns are torch.Tensor or Python scalars. Pure-math
reductions use boolean masks over valid layers rather than torch.nanmean; this
keeps the numeric path clean when `torch.compile` skipped capture for some
layers and filled their slots with NaN.

References:
  - Abnar & Zuidema, 2020 "Quantifying Attention Flow in Transformers"
  - Caron et al., 2021 "Emerging Properties in Self-Supervised Vision
    Transformers" (DINO)  https://github.com/facebookresearch/dino
  - jacobgil ViT-explain
    https://jacobgil.github.io/deeplearning/vision-transformer-explainability
  - Kim et al., ICLR 2025 "See What You Are Told: Visual Attention Sink in
    Large Multimodal Models"  https://arxiv.org/abs/2503.03321

Qwen3-VL patchification constants (PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE)
live in `src.utils.eval_visualizer`; this module imports MERGE_SIZE from there
to avoid duplication.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch

from src.utils.eval_visualizer import MERGE_SIZE


# Default middle-layer range for a 14-layer action expert. Empirically layers
# 3-10 show the strongest spatial focus on task-relevant regions; the final
# few layers disperse attention after the visual signal has been integrated.
# Scale to other expert depths via `compute_middle_layer_range`.
DEFAULT_MIDDLE_LAYER_RANGE_14: tuple[int, int] = (3, 11)  # [start, end)


# ─────────────────────────────────────────────────────────────────────────────
# Layer range
# ─────────────────────────────────────────────────────────────────────────────

def compute_middle_layer_range(
    num_layers: int,
    default: tuple[int, int] = DEFAULT_MIDDLE_LAYER_RANGE_14,
) -> tuple[int, int]:
    """Scale a 14-layer reference range proportionally to an N-layer expert.

    For example: 14 -> (3, 11), 28 -> (6, 22), 7 -> (2, 6).
    """
    start_frac = default[0] / 14
    end_frac = default[1] / 14
    return (
        int(round(num_layers * start_frac)),
        int(round(num_layers * end_frac)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stack & reshape
# ─────────────────────────────────────────────────────────────────────────────

def stack_visual_attention(
    expert_attn: list[list[Any]],
    visual_indices: Any,
    sample_idx: int = 0,
) -> torch.Tensor:
    """Stack sparse expert attention into a dense [S, L, H, A, n_visual] tensor.

    Args:
        expert_attn: nested list `[n_steps][n_layers]` of Tensor `[B, H, A, kv]`
            or None. None entries represent layers for which `torch.compile`
            skipped attention capture.
        visual_indices: positions of visual tokens in the KV sequence. Accepted
            as torch.Tensor, np.ndarray, or list[int].
        sample_idx: which sample in the batch to extract.

    Returns:
        torch.Tensor on CPU, shape `[n_steps, n_layers, n_heads, action_len,
        n_visual]`, float32. Layers that were None get NaN-filled slots so
        that downstream aggregators can mask them out uniformly.

    Raises:
        RuntimeError: if every layer at every step is None (nothing to stack).
    """
    if not expert_attn or not expert_attn[0]:
        raise RuntimeError("expert_attn is empty")
    n_steps = len(expert_attn)
    n_layers = len(expert_attn[0])

    visual_idx_tensor = torch.as_tensor(np.asarray(visual_indices), dtype=torch.long)
    n_visual = int(visual_idx_tensor.numel())

    # Find a reference layer with real weights to infer head/action dims.
    ref: torch.Tensor | None = None
    for s in range(n_steps):
        for lay in expert_attn[s]:
            if lay is not None:
                ref = lay
                break
        if ref is not None:
            break
    if ref is None:
        raise RuntimeError("No expert attention weights captured (all layers None).")

    _, n_heads, action_len, _ = ref.shape
    out = torch.full(
        (n_steps, n_layers, n_heads, action_len, n_visual),
        float("nan"),
        dtype=torch.float32,
    )
    for s in range(n_steps):
        for l in range(n_layers):
            w = expert_attn[s][l]
            if w is None:
                continue
            # w[sample_idx]: [H, A, kv]
            w_sample = w[sample_idx].detach().to(dtype=torch.float32)
            out[s, l] = w_sample[:, :, visual_idx_tensor].cpu()
    return out


def reshape_visual_to_grid(
    attn_flat: Any,
    T_g: int,
    H_g: int,
    W_g: int,
    merge_size: int = MERGE_SIZE,
) -> tuple[torch.Tensor, int, int]:
    """Reshape `[..., n_visual_tokens]` into `[..., T_g, token_H, token_W]`.

    `video_grid_thw` stores `(T_g, H_g, W_g)` where `H_g`/`W_g` are the
    *raw-patch* counts before Qwen3-VL's spatial merge (e.g. 24 for a 384-px
    frame at patch_size=16). The number of visual *tokens* in the language
    sequence is `T_g * (H_g // m) * (W_g // m)` after merge-size grouping.

    The token sequence order is `(t, hm_block, wm_block)` row-major, so a
    plain reshape suffices — each merged token directly corresponds to one
    spatial cell in the (token_H, token_W) grid.

    Args:
        attn_flat: torch.Tensor or np.ndarray, last dim is `n_visual_tokens`.
        T_g, H_g, W_g: entries of `video_grid_thw` (raw patch counts).
        merge_size: Qwen3-VL spatial merge size (default MERGE_SIZE from
            eval_visualizer).

    Returns:
        `(grid, token_H, token_W)` where `grid` is a torch.Tensor of shape
        `[..., T_g, token_H, token_W]` (same leading dims as the input).
    """
    flat = torch.as_tensor(attn_flat)
    m = merge_size if (H_g % merge_size == 0 and W_g % merge_size == 0) else 1
    token_H = H_g // m
    token_W = W_g // m
    expected = T_g * token_H * token_W
    if flat.shape[-1] != expected:
        raise ValueError(
            f"flat dim {flat.shape[-1]} != {T_g}*{token_H}*{token_W}={expected}"
        )
    leading = flat.shape[:-1]
    grid = flat.reshape(*leading, T_g, token_H, token_W)
    return grid, token_H, token_W


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _select_valid_layers(arr: torch.Tensor) -> torch.Tensor:
    """Drop layers whose slice is entirely NaN (compile-skipped capture).

    Input `arr` has shape `[L, H, A, T_g, tH, tW]` (already indexed by ODE
    step). Returns a tensor of shape `[L_valid, H, A, T_g, tH, tW]` containing
    only the layers with at least one finite value.
    """
    if arr.numel() == 0:
        return arr
    # Collapse all dims after L to detect which layers are entirely NaN.
    flat = arr.reshape(arr.shape[0], -1)
    all_nan = torch.isnan(flat).all(dim=1)
    valid = ~all_nan
    if not valid.any():
        return arr[:0]
    return arr[valid]


def aggregate_visual_attention(
    vis_attn_grid: Any,
    strategy: str = "middle_layers_mean_heads",
    ode_step: int | str = 0,
    layer_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Aggregate `[S, L, H, A, T_g, tH, tW]` down to `[T_g, tH, tW]`.

    Strategies:
      - "middle_layers_mean_heads": mean over A, H; mean over layers in
        `layer_range`. **`layer_range` must be passed explicitly** — a None
        value raises ValueError (no silent fallback to the 14-layer default).
      - "mean_layers_heads": mean over A, H, L (all valid layers)
      - "max_heads": mean A; max over H; mean L
      - "last_layer_mean_heads": mean A, H on the last valid layer only

    NaN layers (torch.compile skipped capture) are masked out before
    aggregation via `_select_valid_layers`, so the inner reductions can use
    plain `.mean()` / `.max()` without worrying about NaN propagation.

    Args:
        vis_attn_grid: torch.Tensor or np.ndarray of shape
            `[S, L, H, A, T_g, tH, tW]`.
        strategy: one of the four names above.
        ode_step: int index into the S dim, or "mean" (mean over all steps)
            or "early" (mean over steps 0-2).
        layer_range: `(start, end)` half-open layer range; required for the
            `middle_layers_mean_heads` strategy.

    Returns:
        torch.Tensor of shape `[T_g, tH, tW]`.
    """
    grid = torch.as_tensor(vis_attn_grid)

    if isinstance(ode_step, int):
        step = grid[ode_step]  # [L, H, A, T_g, tH, tW]
    elif ode_step == "mean":
        step = _nanmean_skip_layers(grid, dim=0)
    elif ode_step == "early":
        step = _nanmean_skip_layers(grid[:3], dim=0)
    else:
        raise ValueError(f"bad ode_step {ode_step!r}")

    if strategy == "middle_layers_mean_heads":
        if layer_range is None:
            raise ValueError(
                "strategy='middle_layers_mean_heads' requires an explicit "
                "layer_range; derive one via compute_middle_layer_range(num_layers)."
            )
        L = step.shape[0]
        lo, hi = max(0, layer_range[0]), min(L, layer_range[1])
        if lo >= hi:
            raise ValueError(
                f"empty layer_range {layer_range} for L={L}"
            )
        subset = _select_valid_layers(step[lo:hi])
        if subset.numel() == 0:
            raise RuntimeError(
                f"all layers in range {layer_range} are NaN — nothing to aggregate"
            )
        out = subset.mean(dim=2)  # mean A
        out = out.mean(dim=1)     # mean H
        out = out.mean(dim=0)     # mean layers
    elif strategy == "mean_layers_heads":
        valid = _select_valid_layers(step)
        out = valid.mean(dim=2)
        out = out.mean(dim=1)
        out = out.mean(dim=0)
    elif strategy == "max_heads":
        valid = _select_valid_layers(step)
        out = valid.mean(dim=2)            # mean A
        out = out.amax(dim=1)              # max H
        out = out.mean(dim=0)              # mean L
    elif strategy == "last_layer_mean_heads":
        valid = _select_valid_layers(step)
        if valid.numel() == 0:
            raise RuntimeError("no valid layers to aggregate")
        last = valid[-1]                   # [H, A, T_g, tH, tW]
        out = last.mean(dim=1)             # mean A
        out = out.mean(dim=0)              # mean H
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return out


def _nanmean_skip_layers(grid: torch.Tensor, dim: int) -> torch.Tensor:
    """Step-mean (dim=0) with NaN-skipping via valid-step mask.

    Used by `aggregate_visual_attention` for the "mean" / "early" ode_step
    modes: average along the S dim while ignoring any entries that were
    NaN-filled upstream (e.g., the extremely unlikely case that capture was
    skipped for an entire ODE step). Returns a tensor with `dim` removed.
    """
    mask = ~torch.isnan(grid)
    safe = torch.where(mask, grid, torch.zeros_like(grid))
    total = safe.sum(dim=dim)
    count = mask.sum(dim=dim).clamp(min=1)
    return total / count


def renormalize_visual_subset(attn_grid: torch.Tensor) -> torch.Tensor:
    """Renormalize a non-negative grid so its entries sum to 1.

    Removes the "visual mass vs rest-of-prefix" dilution caused by attention
    sinks (e.g. `<|im_start|>` tokens), exposing the spatial structure inside
    the visual subset. If the input sum is non-positive, the grid is returned
    unchanged (zero grid stays zero).
    """
    grid = torch.as_tensor(attn_grid).float()
    total = grid.sum()
    if total.item() <= 0:
        return grid
    return grid / total


def spatial_entropy(grid: Any) -> float:
    """Normalized Shannon entropy of a non-negative grid.

    Treats the grid as an unnormalized probability mass and returns
    `H / log(N)` where `N = grid.numel()`. Result is in `[0, 1]`:
    0 for perfectly concentrated, 1 for uniform.
    """
    t = torch.as_tensor(grid).double().flatten()
    total = t.sum().clamp(min=1e-12)
    p = t / total
    p = p[p > 0]
    H = -(p * p.log()).sum().item()
    max_H = math.log(t.numel()) if t.numel() > 0 else 0.0
    return float(H / max_H) if max_H > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────

def overlay_attention(
    frame_rgb: np.ndarray,
    attn_hw: Any,
    alpha: float = 0.55,
    upsample: str = "bilinear",
    percentile: tuple[float, float] = (2.0, 98.0),
    cmap_name: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Overlay a low-resolution attention map on an RGB frame.

    Converts `attn_hw` to numpy internally because `cv2.resize` and the
    matplotlib colormap sampling are numpy-native; the RGB frame stays numpy
    throughout.

    Args:
        frame_rgb: uint8 RGB `[H, W, 3]`.
        attn_hw: torch.Tensor or np.ndarray, `[h, w]` low-resolution attention.
        alpha: blending weight for the heatmap (0..1).
        upsample: "bilinear" or "nearest" — cv2 interpolation mode.
        percentile: `(low, high)` percentiles used for contrast stretching
            before applying the colormap. Ignored when both `vmin` and `vmax`
            are provided.
        cmap_name: matplotlib colormap name.
        vmin: explicit lower bound for normalization. When both `vmin` and
            `vmax` are not None, they override `percentile`.
        vmax: explicit upper bound for normalization.

    Returns:
        uint8 RGB `[H, W, 3]`.
    """
    import matplotlib  # local import to keep mpl lazy

    if isinstance(attn_hw, torch.Tensor):
        attn_np = attn_hw.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        attn_np = np.asarray(attn_hw, dtype=np.float32)

    H, W = frame_rgb.shape[:2]
    interp = cv2.INTER_LINEAR if upsample == "bilinear" else cv2.INTER_NEAREST
    attn_up = cv2.resize(attn_np, (W, H), interpolation=interp)
    if vmin is not None and vmax is not None:
        lo, hi = vmin, vmax
    else:
        lo, hi = np.percentile(attn_up, percentile)
    if hi <= lo:
        hi = lo + 1e-8
    attn_norm = np.clip((attn_up - lo) / (hi - lo), 0.0, 1.0)

    cmap = matplotlib.colormaps[cmap_name]
    heatmap = (cmap(attn_norm)[..., :3] * 255.0).astype(np.uint8)
    out = cv2.addWeighted(frame_rgb, 1.0 - alpha, heatmap, alpha, 0.0)
    return out
