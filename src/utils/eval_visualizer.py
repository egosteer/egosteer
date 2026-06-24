"""Evaluation visualization: recover frames from pixel_values, overlay actions, generate HTML report.

This module provides utilities for the evaluate.py script to:
1. Reverse HF Qwen3VL processor normalization to recover displayable images
2. Project 3D actions (wrist + fingertips) onto 2D image frames
3. Generate a self-contained HTML report with metrics and visual overlays
"""

import base64
import json
import pathlib
from datetime import datetime

import cv2
import numpy as np
import torch

from src.dataset.data_transforms import get_absolute_action
from src.utils.geometry import transform_hand_points_from_wrist_to_camera_frame

# CLIP ImageNet normalization constants used by HF Qwen3VL processor.
# Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Qwen3VL vision patchification parameters.
# Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py
TEMPORAL_PATCH_SIZE = 2
PATCH_SIZE = 16
# Spatial merge size: the processor groups patches into merge_size x merge_size blocks
# before flattening, which changes the spatial ordering of patches in pixel_values_videos.
# Source: transformers/models/qwen3_vl/video_processing_qwen3_vl.py
MERGE_SIZE = 2


# ---------------------------------------------------------------------------
# Function A: Recover frames from pixel_values_videos
# ---------------------------------------------------------------------------

def recover_frames_from_pixel_values_videos(
    pixel_values_videos,
    video_grid_thw,
    sample_indices,
    target_size=(384, 384),
):
    """Reverse HF Qwen3VL processor to recover displayable uint8 frames.

    Args:
        pixel_values_videos: Tensor [total_patches, channel_dim] or None.
            Qwen3VL processor output where channel_dim = temporal_patch_size * 3 * patch_size^2.
        video_grid_thw: Tensor [num_videos, 3] with (T_grid, H_grid, W_grid) per video.
        sample_indices: List of batch indices to recover.
        target_size: (H, W) for output images.

    Returns:
        List of uint8 numpy arrays [H, W, 3] (RGB), one per sample_index.
        Returns gray placeholder if pixel_values_videos is None.
    """
    H_target, W_target = target_size

    if pixel_values_videos is None or video_grid_thw is None:
        gray = np.full((H_target, W_target, 3), 128, dtype=np.uint8)
        return [gray.copy() for _ in sample_indices]

    pvv = pixel_values_videos.float().cpu()
    grid = video_grid_thw.long().cpu()
    channel_dim = pvv.shape[1]

    # Validate expected channel_dim = temporal_patch_size * 3 * patch_size^2
    expected_channel_dim = TEMPORAL_PATCH_SIZE * 3 * PATCH_SIZE * PATCH_SIZE
    assert channel_dim == expected_channel_dim, (
        f"Unexpected channel_dim {channel_dim}, expected {expected_channel_dim}. "
        f"Check TEMPORAL_PATCH_SIZE={TEMPORAL_PATCH_SIZE} and PATCH_SIZE={PATCH_SIZE}."
    )

    # Compute per-video patch counts and cumulative offsets
    patch_counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    offsets = [0]
    for cnt in patch_counts:
        offsets.append(offsets[-1] + cnt)

    frames = []
    for idx in sample_indices:
        if idx >= grid.shape[0]:
            frames.append(np.full((H_target, W_target, 3), 128, dtype=np.uint8))
            continue

        T_g, H_g, W_g = grid[idx].tolist()
        start = offsets[idx]
        end = offsets[idx + 1]
        video_patches = pvv[start:end]  # [T_g * H_g * W_g, channel_dim]

        # Reverse the HF Qwen3VL processor patchification.
        # The processor permutes to [batch, T_g, H_g//m, W_g//m, m_h, m_w, C, tp, ph, pw]
        # then flattens to [batch, T_g * H_g * W_g, C * tp * ph * pw].
        # Channel dim order is [C, tp, ph, pw] (channel outermost).
        # Source: transformers/models/qwen3_vl/video_processing_qwen3_vl.py
        m = MERGE_SIZE if (H_g % MERGE_SIZE == 0 and W_g % MERGE_SIZE == 0) else 1
        video_patches = video_patches.reshape(
            T_g, H_g // m, W_g // m, m, m, 3, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,
        )
        # Un-merge spatial blocks: interleave merge dims back into spatial dims
        # [T_g, H_g//m, W_g//m, m_h, m_w, C, tp, ph, pw]
        #  → permute → [T_g, H_g//m, m_h, W_g//m, m_w, C, tp, ph, pw]
        #  → reshape → [T_g, H_g, W_g, C, tp, ph, pw]
        video_patches = video_patches.permute(0, 1, 3, 2, 4, 5, 6, 7, 8).reshape(
            T_g, H_g, W_g, 3, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,
        )

        # Extract last frame: last temporal group, second frame in the pair
        last_t = T_g - 1
        last_frame_patches = video_patches[last_t, :, :, :, -1, :, :]  # [H_g, W_g, 3, 14, 14]

        # Rearrange patches to image: [H_g, W_g, 3, 14, 14] -> [3, H_g*14, W_g*14]
        # permute to [3, H_g, 14, W_g, 14] then reshape
        frame = last_frame_patches.permute(2, 0, 3, 1, 4).reshape(3, H_g * PATCH_SIZE, W_g * PATCH_SIZE)

        # CHW -> HWC
        frame = frame.permute(1, 2, 0).numpy()  # [H_g*14, W_g*14, 3]

        # Denormalize: pixel = pixel * std + mean
        frame = frame * CLIP_STD + CLIP_MEAN
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

        # Resize to target if spatial dims differ
        h, w = frame.shape[:2]
        if (h, w) != (H_target, W_target):
            frame = cv2.resize(frame, (W_target, H_target), interpolation=cv2.INTER_LINEAR)

        frames.append(frame)

    return frames


def recover_all_frames(
    pixel_values_videos,
    video_grid_thw,
    sample_idx: int,
    target_size=(384, 384),
):
    """Recover ALL frames (not just the last) for one sample's video.

    Mirrors :func:`recover_frames_from_pixel_values_videos` but iterates over
    every ``(temporal_group, frame_in_group)`` pair instead of extracting only
    the most-recent frame. Useful for attention-visualization workflows that
    need every frame in the observation history, not just the last one.

    See also :func:`recover_frames_from_pixel_values_videos`, which returns
    only the last frame per sample (more efficient when that is all you need).

    Args:
        pixel_values_videos: Tensor [total_patches, channel_dim].
        video_grid_thw: Tensor [num_videos, 3] with (T_grid, H_grid, W_grid).
        sample_idx: Which sample in the batch to recover.
        target_size: (H, W) output size.

    Returns:
        (frames, grid) where `frames` is a list of uint8 RGB arrays of length
        ``T_g * TEMPORAL_PATCH_SIZE``, and `grid` is ``(T_g, H_g, W_g)``.
    """
    pvv = pixel_values_videos.float().cpu()
    grid = video_grid_thw.long().cpu()
    channel_dim = pvv.shape[1]

    expected_channel_dim = TEMPORAL_PATCH_SIZE * 3 * PATCH_SIZE * PATCH_SIZE
    assert channel_dim == expected_channel_dim, (
        f"Unexpected channel_dim {channel_dim}, expected {expected_channel_dim}"
    )

    # Per-sample offsets into the packed patch sequence.
    patch_counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    offsets = [0]
    for cnt in patch_counts:
        offsets.append(offsets[-1] + cnt)

    T_g, H_g, W_g = grid[sample_idx].tolist()
    start, end = offsets[sample_idx], offsets[sample_idx + 1]
    video_patches = pvv[start:end]  # [T_g * H_g * W_g, channel_dim]

    m = MERGE_SIZE if (H_g % MERGE_SIZE == 0 and W_g % MERGE_SIZE == 0) else 1
    video_patches = video_patches.reshape(
        T_g, H_g // m, W_g // m, m, m, 3, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,
    )
    video_patches = video_patches.permute(0, 1, 3, 2, 4, 5, 6, 7, 8).reshape(
        T_g, H_g, W_g, 3, TEMPORAL_PATCH_SIZE, PATCH_SIZE, PATCH_SIZE,
    )
    # video_patches: [T_g, H_g, W_g, 3, tp, ph, pw]

    H_target, W_target = target_size
    frames = []
    for t_idx in range(T_g):
        for tp_idx in range(TEMPORAL_PATCH_SIZE):
            fp = video_patches[t_idx, :, :, :, tp_idx, :, :]  # [H_g, W_g, 3, ph, pw]
            frame = fp.permute(2, 0, 3, 1, 4).reshape(
                3, H_g * PATCH_SIZE, W_g * PATCH_SIZE,
            )
            frame = frame.permute(1, 2, 0).numpy()
            frame = frame * CLIP_STD + CLIP_MEAN
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            h, w = frame.shape[:2]
            if (h, w) != (H_target, W_target):
                frame = cv2.resize(frame, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

    return frames, (T_g, H_g, W_g)


# ---------------------------------------------------------------------------
# Function B: Prepare visualization sample
# ---------------------------------------------------------------------------

def prepare_vis_sample(
    batch,
    frame,
    pred_actions,
    gt_actions,
    sample_idx,
    normalizer,
    use_relative_action,
    global_index,
):
    """Process one sample from a batch into a visualization-ready dict.

    Args:
        batch: Collated batch dict (CPU tensors).
        frame: Recovered uint8 image [H, W, 3] (RGB).
        pred_actions: Unnormalized predicted actions [H_action, 48].
        gt_actions: Unnormalized GT actions [H_action, 48].
        sample_idx: Index within this batch.
        normalizer: LinearNormalizer with 'states'/'actions'/'motions' keys.
        use_relative_action: Whether actions are relative to current state.
        global_index: Global sample counter across all batches.

    Returns:
        Dict with keys: frame, pred_wrist, pred_hand_cam, gt_wrist, gt_hand_cam,
        intrinsic, sample_mae, global_index. All numpy arrays.
    """
    pred = pred_actions.float()
    gt = gt_actions.float()

    # Convert relative -> absolute if needed
    if use_relative_action:
        n_states = int(batch["n_states"][sample_idx].item())
        state_normalized = batch["states"][sample_idx, n_states - 1, :].float()
        current_state = normalizer["states"].unnormalize(state_normalized.unsqueeze(0)).squeeze(0)
        pred = get_absolute_action(current_state, pred)
        gt = get_absolute_action(current_state, gt)

    # Split wrist (18D) and hand (30D)
    pred_wrist = pred[:, :18]
    pred_hand = pred[:, 18:48]
    gt_wrist = gt[:, :18]
    gt_hand = gt[:, 18:48]

    # Transform hand points from wrist frame to camera frame
    pred_hand_cam = transform_hand_points_from_wrist_to_camera_frame(
        pred_hand.numpy().astype(np.float32),
        pred_wrist.numpy().astype(np.float32),
    )
    gt_hand_cam = transform_hand_points_from_wrist_to_camera_frame(
        gt_hand.numpy().astype(np.float32),
        gt_wrist.numpy().astype(np.float32),
    )

    # Head intrinsic [fx, fy, cx, cy]. camera_intrinsic is flat [total_slots, 4]
    # keyed to rendered <camera> tokens in token mode, so it no longer admits
    # a per-sample index; use the raw per-sample intrinsic here.
    if "intrinsic" not in batch:
        raise ValueError("'intrinsic' missing from batch; ensure it is in collatable_keys.")
    intrinsic = batch["intrinsic"][sample_idx].float().cpu().numpy()

    # Per-step valid mask: True for non-padded timesteps.
    # actions_valid_mask shape: [H_action, 48] — check first dim of any column.
    step_valid = batch["actions_valid_mask"][sample_idx, :, 0].bool().numpy()  # [H_action]

    # Per-sample MAE (on valid steps only)
    valid_mask = batch["actions_valid_mask"][sample_idx].bool()
    mae = torch.mean(torch.abs(pred_actions[valid_mask] - gt_actions[valid_mask])).item()

    return {
        "frame": frame,
        "pred_wrist": pred_wrist.numpy(),
        "pred_hand_cam": pred_hand_cam if isinstance(pred_hand_cam, np.ndarray) else pred_hand_cam.numpy(),
        "gt_wrist": gt_wrist.numpy(),
        "gt_hand_cam": gt_hand_cam if isinstance(gt_hand_cam, np.ndarray) else gt_hand_cam.numpy(),
        "intrinsic": intrinsic,
        "step_valid": step_valid,
        "sample_mae": mae,
        "global_index": global_index,
    }


# ---------------------------------------------------------------------------
# Function C: Render overlay image
# ---------------------------------------------------------------------------

def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    """Perspective projection from 3D camera-frame points to 2D pixel coordinates.

    Args:
        points_3d: np.ndarray [N, 3] in camera frame.
        fx, fy: Focal lengths.
        cx, cy: Principal point.

    Returns:
        np.ndarray [N, 2] pixel coordinates (u, v).

    # Source: visualize.py#L499-L511
    """
    z = np.clip(points_3d[:, 2], 1e-6, None)
    u = fx * (points_3d[:, 0] / z) + cx
    v = fy * (points_3d[:, 1] / z) + cy
    return np.stack([u, v], axis=1)


def lerp_color(color_start, color_end, t):
    """Linear interpolation between two BGR color tuples, t in [0, 1]."""
    return tuple(int(s + (e - s) * t) for s, e in zip(color_start, color_end))


def render_overlay_image(vis_sample, action_stride=4):
    """Draw pred/GT action keypoints on a frame with time-varying colors and colorbar.

    Args:
        vis_sample: Dict from prepare_vis_sample.
        action_stride: Draw every N-th action timestep.

    Returns:
        BGR uint8 numpy array with overlay and colorbar.
    """
    # Convert RGB frame to BGR for cv2 drawing
    frame_rgb = vis_sample["frame"]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    canvas = frame_bgr.copy()
    H, W = canvas.shape[:2]

    intrinsic = vis_sample["intrinsic"]
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]

    pred_wrist = vis_sample["pred_wrist"]    # [T, 18]
    pred_hand = vis_sample["pred_hand_cam"]  # [T, 30]
    gt_wrist = vis_sample["gt_wrist"]        # [T, 18]
    gt_hand = vis_sample["gt_hand_cam"]      # [T, 30]

    step_valid = vis_sample["step_valid"]  # [T] boolean
    total_steps = pred_wrist.shape[0]
    # Only select valid (non-padded) timesteps at the given stride
    valid_steps = [t for t in range(total_steps) if step_valid[t]]
    if not valid_steps:
        return _append_colorbar(
            canvas, total_steps,
            GT_COLOR_START, GT_COLOR_END, PRED_COLOR_START, PRED_COLOR_END,
        )
    timesteps = valid_steps[::action_stride]
    if valid_steps[-1] not in timesteps:
        timesteps.append(valid_steps[-1])
    num_valid = len(valid_steps)

    # Color gradients (BGR): cross-hue for maximum contrast between t=0 and t=T
    GT_COLOR_START = (255, 100, 0)      # bright blue
    GT_COLOR_END = (200, 255, 0)        # cyan
    PRED_COLOR_START = (50, 50, 255)    # bright red
    PRED_COLOR_END = (0, 220, 255)      # yellow

    # Draw trajectory lines first (thinner, behind points)
    max_valid_t = valid_steps[-1]
    for seq_i in range(len(timesteps) - 1):
        t0 = timesteps[seq_i]
        t1 = timesteps[seq_i + 1]
        ratio0 = t0 / max(max_valid_t, 1)
        ratio1 = t1 / max(max_valid_t, 1)

        for hand_idx in range(2):
            # GT wrist trajectory
            gt_pos0 = gt_wrist[t0, hand_idx * 3: hand_idx * 3 + 3].reshape(1, 3)
            gt_pos1 = gt_wrist[t1, hand_idx * 3: hand_idx * 3 + 3].reshape(1, 3)
            gt_2d_0 = project_3d_to_2d(gt_pos0, fx, fy, cx, cy)[0]
            gt_2d_1 = project_3d_to_2d(gt_pos1, fx, fy, cx, cy)[0]
            gt_color = lerp_color(GT_COLOR_START, GT_COLOR_END, (ratio0 + ratio1) / 2)
            cv2.line(canvas, _pt(gt_2d_0), _pt(gt_2d_1), gt_color, 2, cv2.LINE_AA)

            # Pred wrist trajectory
            pred_pos0 = pred_wrist[t0, hand_idx * 3: hand_idx * 3 + 3].reshape(1, 3)
            pred_pos1 = pred_wrist[t1, hand_idx * 3: hand_idx * 3 + 3].reshape(1, 3)
            pred_2d_0 = project_3d_to_2d(pred_pos0, fx, fy, cx, cy)[0]
            pred_2d_1 = project_3d_to_2d(pred_pos1, fx, fy, cx, cy)[0]
            pred_color = lerp_color(PRED_COLOR_START, PRED_COLOR_END, (ratio0 + ratio1) / 2)
            cv2.line(canvas, _pt(pred_2d_0), _pt(pred_2d_1), pred_color, 2, cv2.LINE_AA)

    # Draw keypoints at each selected timestep
    for t in timesteps:
        ratio = t / max(max_valid_t, 1)
        gt_color = lerp_color(GT_COLOR_START, GT_COLOR_END, ratio)
        pred_color = lerp_color(PRED_COLOR_START, PRED_COLOR_END, ratio)

        for hand_idx in range(2):
            # Wrist position: 3D -> 2D
            gt_wrist_3d = gt_wrist[t, hand_idx * 3: hand_idx * 3 + 3].reshape(1, 3)
            pred_wrist_3d = pred_wrist[t, hand_idx * 3: hand_idx * 3 + 3].reshape(1, 3)
            gt_w2d = project_3d_to_2d(gt_wrist_3d, fx, fy, cx, cy)[0]
            pred_w2d = project_3d_to_2d(pred_wrist_3d, fx, fy, cx, cy)[0]

            # Fingertips: 5 points per hand, each 3D
            hand_offset = hand_idx * 15
            gt_fingers = gt_hand[t, hand_offset: hand_offset + 15].reshape(5, 3)
            pred_fingers = pred_hand[t, hand_offset: hand_offset + 15].reshape(5, 3)
            gt_f2d = project_3d_to_2d(gt_fingers, fx, fy, cx, cy)    # [5, 2]
            pred_f2d = project_3d_to_2d(pred_fingers, fx, fy, cx, cy)  # [5, 2]

            # Draw GT: wrist circle + lines to fingertips + fingertip circles
            _draw_hand(canvas, gt_w2d, gt_f2d, gt_color, radius=3)
            # Draw Pred: same structure
            _draw_hand(canvas, pred_w2d, pred_f2d, pred_color, radius=3)

    # Add text overlay: sample index and MAE
    label = f"#{vis_sample['global_index']}  MAE: {vis_sample['sample_mae']:.4f}"
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Append colorbar on the right side (show valid range)
    canvas = _append_colorbar(
        canvas, num_valid,
        GT_COLOR_START, GT_COLOR_END,
        PRED_COLOR_START, PRED_COLOR_END,
        bar_width=36,
    )

    return canvas


def _pt(coord_2d):
    """Convert float 2D coordinate to integer tuple for cv2."""
    return (int(round(coord_2d[0])), int(round(coord_2d[1])))


def _draw_hand(canvas, wrist_2d, fingers_2d, color, radius=3):
    """Draw one hand's wrist + fingertip keypoints with connections."""
    H, W = canvas.shape[:2]
    margin = 50

    # Skip if wrist is far out of bounds
    if not (-margin <= wrist_2d[0] <= W + margin and -margin <= wrist_2d[1] <= H + margin):
        return

    wp = _pt(wrist_2d)
    # Black outline then filled color for visibility when overlapping
    cv2.circle(canvas, wp, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, wp, radius + 1, color, -1, cv2.LINE_AA)

    for i in range(fingers_2d.shape[0]):
        fp = _pt(fingers_2d[i])
        if -margin <= fp[0] <= W + margin and -margin <= fp[1] <= H + margin:
            cv2.line(canvas, wp, fp, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, fp, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, fp, radius, color, -1, cv2.LINE_AA)


def _append_colorbar(canvas, total_steps, gt_start, gt_end, pred_start, pred_end, bar_width=36):
    """Append a vertical timestep colorbar to the right of the image.

    Left half = GT gradient (green), right half = Pred gradient (orange).
    """
    H, W = canvas.shape[:2]
    bar = np.zeros((H, bar_width, 3), dtype=np.uint8)

    # Vertical gradient region (with top/bottom margin for labels)
    margin_top = 20
    margin_bottom = 20
    grad_h = H - margin_top - margin_bottom

    if grad_h > 0:
        half_w = bar_width // 2
        for y in range(grad_h):
            ratio = y / max(grad_h - 1, 1)
            gt_c = lerp_color(gt_start, gt_end, ratio)
            pred_c = lerp_color(pred_start, pred_end, ratio)
            bar[margin_top + y, :half_w] = gt_c
            bar[margin_top + y, half_w:] = pred_c

    # Labels
    cv2.putText(bar, "t=0", (2, margin_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        bar, f"t={total_steps - 1}", (1, H - margin_bottom + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return np.concatenate([canvas, bar], axis=1)


# ---------------------------------------------------------------------------
# Function D: Generate HTML report
# ---------------------------------------------------------------------------

def encode_image_to_base64(img_bgr):
    """Encode a BGR numpy array as a base64 PNG data URI."""
    _, buf = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


def encode_file_to_base64(path):
    """Encode a file as a base64 data URI (auto-detect PNG/JPEG)."""
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def generate_html_report(
    output_dir,
    eval_config,
    train_config_path,
    checkpoint_path,
    metrics_json_path,
    metric_png_paths,
    vis_images,
):
    """Generate a self-contained HTML report with base64-embedded images.

    Args:
        output_dir: pathlib.Path for output directory.
        eval_config: Dict of evaluation config.
        train_config_path: Path string to training config.
        checkpoint_path: Path string to checkpoint.
        metrics_json_path: Path to metrics.json.
        metric_png_paths: List of paths to metric PNG files.
        vis_images: List of (global_index, sample_mae, bgr_image) tuples.

    Returns:
        Path to generated report.html.
    """
    output_dir = pathlib.Path(output_dir)

    # Load metrics
    metrics = {}
    if metrics_json_path.exists():
        with open(metrics_json_path) as f:
            metrics = json.load(f)

    # Sort vis images by MAE descending (worst first)
    vis_images_sorted = sorted(vis_images, key=lambda x: x[1], reverse=True)

    # Build HTML sections
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ckpt_name = pathlib.Path(checkpoint_path).stem

    # Config table rows
    config_rows = _build_config_rows(eval_config, train_config_path, checkpoint_path)

    # Metrics cards + detail table
    metrics_cards_html = _build_metrics_cards(metrics)
    metrics_detail_html = _build_metrics_detail_table(metrics)

    # Metric plot images
    metric_plots_html = ""
    for png_path in metric_png_paths:
        b64 = encode_file_to_base64(png_path)
        name = pathlib.Path(png_path).stem.replace("_", " ").title()
        metric_plots_html += f'''
        <div class="plot-item">
            <img src="{b64}" alt="{name}" loading="lazy">
            <p class="plot-label">{name}</p>
        </div>'''

    # Visualization grid
    vis_grid_html = ""
    for idx, mae, img_bgr in vis_images_sorted:
        b64 = encode_image_to_base64(img_bgr)
        vis_grid_html += f'''
        <div class="vis-item">
            <img src="{b64}" alt="Sample {idx}" loading="lazy">
            <div class="vis-caption">
                <span class="vis-idx">#{idx}</span>
                <span class="vis-mae">MAE: {mae:.4f}</span>
            </div>
        </div>'''

    html = _HTML_TEMPLATE.format(
        title=f"Eval Report — {ckpt_name}",
        timestamp=timestamp,
        config_rows=config_rows,
        metrics_cards=metrics_cards_html,
        metrics_detail=metrics_detail_html,
        metric_plots=metric_plots_html,
        vis_grid=vis_grid_html,
        total_samples=len(vis_images),
    )

    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ---------------------------------------------------------------------------
# HTML template and helpers
# ---------------------------------------------------------------------------

def _build_config_rows(eval_config, train_config_path, checkpoint_path):
    rows = [
        ("Checkpoint", checkpoint_path),
        ("Train Config", train_config_path),
        ("Flow Steps", eval_config.get("flow_steps", "default")),
        ("Inference Delay", eval_config.get("inference_delay", 0)),
        ("Num Shards", eval_config.get("num_shards", "—")),
        ("Num Samples", eval_config.get("num_samples", "—")),
        ("Batch Size", eval_config.get("batch_size", "—")),
        ("Device", eval_config.get("device", "—")),
        ("Seed", eval_config.get("seed", "—")),
    ]
    return "\n".join(
        f'<tr><td class="cfg-key">{k}</td><td class="cfg-val">{v}</td></tr>'
        for k, v in rows
    )


def _build_metrics_cards(metrics):
    """Build hero metric cards for key indicators."""
    cards = [
        ("Overall MAE", metrics.get("overall_mae"), "{:.4f}", "lower is better"),
        ("Acc @0.1", (metrics.get("action_accuracy") or {}).get("threshold_0.1"), "{:.2%}", "higher is better"),
        ("Acc @0.2", (metrics.get("action_accuracy") or {}).get("threshold_0.2"), "{:.2%}", "higher is better"),
        ("Endpoint Error", (metrics.get("trajectory") or {}).get("endpoint_error_mean"), "{:.4f}", "lower is better"),
        ("1st Diff Error", metrics.get("first_diff_error"), "{:.4f}", "lower is better"),
    ]
    html = ""
    for label, value, fmt, hint in cards:
        if value is not None:
            formatted = fmt.format(value)
        else:
            formatted = "N/A"
        html += f'''
        <div class="metric-card">
            <div class="metric-value">{formatted}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-hint">{hint}</div>
        </div>'''
    return html


def _build_metrics_detail_table(metrics):
    """Build a detailed metrics table with all available metrics."""
    if not metrics:
        return "<p>No metrics available.</p>"

    rows = []

    def add_row(key, value, fmt="{:.6f}"):
        if value is not None:
            rows.append(f'<tr><td>{key}</td><td>{fmt.format(value)}</td></tr>')

    add_row("Overall MAE", metrics.get("overall_mae"))
    add_row("1st Order Diff Error", metrics.get("first_diff_error"))
    add_row("2nd Order Diff Error", metrics.get("second_diff_error"))

    acc = metrics.get("action_accuracy") or {}
    for t_key, t_val in sorted(acc.items()):
        add_row(f"Action Accuracy ({t_key})", t_val, "{:.4f}")

    traj = metrics.get("trajectory") or {}
    add_row("Endpoint Error (mean)", traj.get("endpoint_error_mean"))
    add_row("Endpoint Error (std)", traj.get("endpoint_error_std"))
    add_row("Traj Length Error (mean)", traj.get("trajectory_length_error_mean"))
    add_row("Traj Length Error (std)", traj.get("trajectory_length_error_std"))
    add_row("Mean Error (mean)", traj.get("mean_error_mean"))
    add_row("Max Error (mean)", traj.get("max_error_mean"))

    loss = metrics.get("loss_over_time") or {}
    add_row("L1 Loss (mean over time)", loss.get("l1_mean"))
    add_row("L2 Loss (mean over time)", loss.get("l2_mean"))

    smooth = metrics.get("smoothness") or {}
    add_row("Smoothness 1st Diff", smooth.get("first_diff_error"))
    add_row("Smoothness 2nd Diff", smooth.get("second_diff_error"))

    return "\n".join(rows)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
    --bg: #f5f6fa;
    --card-bg: #ffffff;
    --text: #2d3436;
    --text-secondary: #636e72;
    --border: #dfe6e9;
    --accent: #0984e3;
    --green: #00b894;
    --orange: #e17055;
    --shadow: 0 2px 8px rgba(0,0,0,0.08);
    --radius: 10px;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 24px;
}}
.container {{ max-width: 1440px; margin: 0 auto; }}

/* Header */
.header {{
    background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
    color: #fff;
    padding: 32px 40px;
    border-radius: var(--radius);
    margin-bottom: 24px;
    box-shadow: var(--shadow);
}}
.header h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
.header .subtitle {{ opacity: 0.7; font-size: 0.9rem; }}

/* Section */
.section {{
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
}}
.section h2 {{
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
    color: var(--text);
}}

/* Config table */
.cfg-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
.cfg-table td {{ padding: 6px 12px; border-bottom: 1px solid var(--border); }}
.cfg-key {{ font-weight: 600; width: 180px; color: var(--text-secondary); }}
.cfg-val {{ font-family: 'SF Mono', 'Fira Code', monospace; word-break: break-all; }}

/* Metric cards */
.metrics-cards {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}}
.metric-card {{
    flex: 1;
    min-width: 150px;
    background: var(--bg);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid var(--border);
}}
.metric-value {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
.metric-label {{ font-size: 0.85rem; font-weight: 600; color: var(--text); margin-top: 4px; }}
.metric-hint {{ font-size: 0.72rem; color: var(--text-secondary); margin-top: 2px; }}

/* Detail table (collapsible) */
details {{ margin-top: 8px; }}
details summary {{
    cursor: pointer;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--accent);
    padding: 4px 0;
}}
.detail-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.85rem; }}
.detail-table td {{ padding: 5px 12px; border-bottom: 1px solid var(--border); }}
.detail-table td:first-child {{ font-weight: 500; width: 260px; color: var(--text-secondary); }}
.detail-table td:last-child {{ font-family: 'SF Mono', 'Fira Code', monospace; }}

/* Metric plots */
.plots-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}}
.plot-item {{
    text-align: center;
    background: var(--bg);
    border-radius: 8px;
    padding: 12px;
    border: 1px solid var(--border);
}}
.plot-item img {{
    width: 100%;
    border-radius: 6px;
    cursor: pointer;
    transition: transform 0.2s ease;
}}
.plot-item img:hover {{ transform: scale(1.02); }}
.plot-label {{ font-size: 0.82rem; color: var(--text-secondary); margin-top: 6px; }}

/* Visualization grid */
.vis-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}
.vis-count {{ font-size: 0.85rem; color: var(--text-secondary); }}
.vis-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}}
.vis-item {{
    background: var(--bg);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}}
.vis-item:hover {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}}
.vis-item img {{
    width: 100%;
    display: block;
    cursor: pointer;
}}
.vis-caption {{
    padding: 6px 10px;
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
}}
.vis-idx {{ font-weight: 600; color: var(--text); }}
.vis-mae {{ font-family: 'SF Mono', 'Fira Code', monospace; color: var(--orange); }}

/* Lightbox */
.lightbox {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85);
    z-index: 1000;
    justify-content: center;
    align-items: center;
    cursor: pointer;
}}
.lightbox.active {{ display: flex; }}
.lightbox img {{
    max-width: 90vw;
    max-height: 90vh;
    border-radius: 8px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}}

/* Responsive */
@media (max-width: 1200px) {{
    .vis-grid {{ grid-template-columns: repeat(3, 1fr); }}
}}
@media (max-width: 800px) {{
    .vis-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .plots-grid {{ grid-template-columns: 1fr; }}
    .metrics-cards {{ flex-direction: column; }}
}}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>{title}</h1>
    <div class="subtitle">Generated: {timestamp}</div>
</div>

<div class="section">
    <h2>Configuration</h2>
    <table class="cfg-table">
        {config_rows}
    </table>
</div>

<div class="section">
    <h2>Metrics Overview</h2>
    <div class="metrics-cards">
        {metrics_cards}
    </div>
    <details>
        <summary>Show all metrics</summary>
        <table class="detail-table">
            {metrics_detail}
        </table>
    </details>
</div>

<div class="section">
    <h2>Metric Plots</h2>
    <div class="plots-grid">
        {metric_plots}
    </div>
</div>

<div class="section">
    <div class="vis-header">
        <h2 style="border: none; margin: 0; padding: 0;">2D Action Visualization</h2>
        <span class="vis-count">{total_samples} samples (sorted by MAE, worst first)</span>
    </div>
    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 12px;">
        <span style="color: #00b894;">&#9679; Green = Ground Truth</span> &nbsp;
        <span style="color: #e17055;">&#9679; Orange = Prediction</span> &nbsp;
        Colors darken with increasing timestep. Click images to enlarge.
    </p>
    <div class="vis-grid">
        {vis_grid}
    </div>
</div>

</div>

<!-- Lightbox for image zoom -->
<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
    <img id="lightbox-img" src="" alt="Enlarged">
</div>

<script>
document.querySelectorAll('.vis-item img, .plot-item img').forEach(function(img) {{
    img.addEventListener('click', function() {{
        var lb = document.getElementById('lightbox');
        document.getElementById('lightbox-img').src = this.src;
        lb.classList.add('active');
    }});
}});
</script>
</body>
</html>"""
