"""Offline dataset evaluation for EgoSteer flow matching.

Loads a trained checkpoint and its corresponding training config, samples from
WebDataset shards, runs flow inference, and computes action metrics.

Usage:
    python evaluate.py \
        checkpoint_path=/path/to/ckpt.pt \
        train_config_path=/path/to/.hydra/config.yaml

    # With RTC prefix condition (pin first 4 GT steps):
    python evaluate.py \
        checkpoint_path=/path/to/ckpt.pt \
        train_config_path=/path/to/.hydra/config.yaml \
        inference_delay=4
"""

import pathlib
import pickle
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.checkpoint_util import load_checkpoint
from src.utils.eval_visualizer import (
    recover_frames_from_pixel_values_videos,
    prepare_vis_sample,
    render_overlay_image,
    generate_html_report,
)

# Register the eval resolver used in training configs.
# Saved configs are normally fully resolved, but this is a safety fallback.
OmegaConf.register_new_resolver("eval", eval, replace=True)

torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_normalizer(path):
    """Load a pre-computed normalizer pickle."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Normalizer not found: {path}. "
            "A normalizer is required for evaluation."
        )
    with open(path, "rb") as f:
        normalizer = pickle.load(f)
    print(f"Loaded normalizer from {path}")
    return normalizer


def build_eval_inputs(batch, device, dtype):
    """Extract model inputs from a collated batch for flow inference.

    Mirrors EgoSteerInference.build_model_inputs (flow mode branch).
    # Source: src/policy/egosteer_inference_wrapper.py#L260-L288
    """
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pixel_values": (
            batch["pixel_values"].to(dtype=dtype, device=device)
            if batch["pixel_values"] is not None else None
        ),
        "image_grid_thw": (
            batch["image_grid_thw"].to(device)
            if batch["image_grid_thw"] is not None else None
        ),
        "pixel_values_videos": (
            batch["pixel_values_videos"].to(dtype=dtype, device=device)
            if batch["pixel_values_videos"] is not None else None
        ),
        "video_grid_thw": (
            batch["video_grid_thw"].to(device)
            if batch["video_grid_thw"] is not None else None
        ),
        "mm_token_type_ids": batch["mm_token_type_ids"].to(device),
        "states": batch["states"].to(dtype=dtype, device=device),
        "n_states": batch["n_states"].to(device),
        "n_actions": batch["n_actions"].to(dtype=torch.long, device=device),
        "is_vla_data": batch["is_vla_data"].to(device),
        # Flow-specific fields
        "answer_start_idx": batch["answer_start_idx"].to(device),
        "actions": batch["actions"].to(dtype=dtype, device=device),
        "actions_valid_mask": batch["actions_valid_mask"].to(dtype=torch.bool, device=device),
    }
    if "camera_intrinsic" in batch:
        inputs["camera_intrinsic"] = batch["camera_intrinsic"].to(dtype=dtype, device=device)
    return inputs


def build_head_video_indices(batch, video_grid_thw):
    """Map batch sample index -> head video entry index in video_grid_thw.

    Collator flattens per-sample videos in sample order. For dual-view VLA
    samples the order is [head, chest] per sample, so head indices are 0,2,4...
    For single-view samples the head index matches sample index.
    """
    if video_grid_thw is None:
        return None

    batch_size = int(batch["input_ids"].shape[0])
    num_video_entries = int(video_grid_thw.shape[0])

    if num_video_entries == batch_size:
        return list(range(batch_size))

    if num_video_entries == batch_size * 2:
        return [i * 2 for i in range(batch_size)]

    # Fallback for mixed/irregular batches: best-effort alignment for first B entries.
    # This keeps report generation alive while making ambiguity explicit in logs.
    print(
        "WARNING: Ambiguous video entry layout for visualization "
        f"(batch_size={batch_size}, num_video_entries={num_video_entries}). "
        "Falling back to first B video entries."
    )
    return list(range(min(batch_size, num_video_entries)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("src", "config")),
    config_name="eval_config",
)
def main(eval_cfg):
    OmegaConf.resolve(eval_cfg)
    device = torch.device(eval_cfg.device)
    dtype = torch.bfloat16
    rng = np.random.default_rng(eval_cfg.seed)
    output_dir = pathlib.Path(eval_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load training config from the previous run ----
    train_cfg = OmegaConf.load(eval_cfg.train_config_path)
    print(f"Loaded training config from {eval_cfg.train_config_path}")

    # ---- 2. Instantiate model and load checkpoint ----
    model = hydra.utils.instantiate(train_cfg.policy)
    load_checkpoint(model, eval_cfg.checkpoint_path)
    model.to(dtype=dtype).eval().to(device)

    if eval_cfg.flow_steps is not None:
        model.num_inference_steps = int(eval_cfg.flow_steps)
        print(f"Overriding flow inference steps to {eval_cfg.flow_steps}")

    print(f"Model on {device}, dtype={dtype}, "
          f"flow_steps={model.num_inference_steps}, "
          f"action_dim={model.action_dim}")

    # ---- 3. Load normalizer ----
    # Provided explicitly via eval config, not read from the training config
    normalizer = load_normalizer(eval_cfg.normalizer_path)
    use_relative_action = bool(train_cfg.dataset.vla_dataset.use_relative_action)
    norm_key = "actions" if use_relative_action else "motions"

    # ---- 4. Instantiate collator (override mode to "infer" for prompt-only encoding) ----
    collator = hydra.utils.instantiate(train_cfg.data_collator, mode="infer")
    depth_clip_range = OmegaConf.to_container(
        train_cfg.data.depth_clip_range, resolve=True,
    ) if train_cfg.data.get("depth_clip_range") else None

    # ---- 5. Select shards and build dataset ----
    from src.dataset.wds_dataset import expand_shard_patterns
    from src.dataset.vla_dataset import VLAWdsDataset

    all_shards = []
    for ds_cfg in eval_cfg.eval_datasets:
        urls, _ = expand_shard_patterns(ds_cfg["shard_urls"])
        all_shards.extend(urls)

    if not all_shards:
        print("ERROR: No shards found from eval_datasets. Check shard_urls paths.")
        sys.exit(1)

    num_shards = min(eval_cfg.num_shards, len(all_shards))
    selected_indices = rng.choice(len(all_shards), size=num_shards, replace=False)
    selected_shards = [all_shards[i] for i in selected_indices]
    print(f"Selected {num_shards}/{len(all_shards)} shards:")
    for s in selected_shards:
        print(f"  {s}")

    vla_ds_cfg = train_cfg.dataset.vla_dataset
    shape_meta = OmegaConf.to_container(train_cfg.data.shape_meta, resolve=True)
    target_image_size = train_cfg.data.get("target_image_size")
    load_depth = bool(vla_ds_cfg.get("load_depth", False))
    load_chest = bool(vla_ds_cfg.get("load_chest", False))

    dataset = VLAWdsDataset(
        wds_datasets=[{"shard_urls": selected_shards, "weight": 1.0, "name": "eval"}],
        shape_meta=shape_meta,
        use_relative_action=vla_ds_cfg.use_relative_action,
        mode="val",
        shuffle_buffer=0,
        depth_clip_range=depth_clip_range,
        video_base_fps=float(train_cfg.data.video_base_fps),
        target_image_size=list(target_image_size) if target_image_size else None,
        load_depth=load_depth,
        load_chest=load_chest,
    )
    dataset.set_collator(collator)
    dataset.set_normalizer(normalizer)

    # ---- 6. DataLoader + inference loop ----
    # Use the collator directly (not dataset.get_collator()) to keep mode="infer".
    # get_collator() would override mode to the dataset's "val", encoding the full
    # conversation into input_ids instead of prompt-only.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_cfg.batch_size,
        num_workers=eval_cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    inference_delay = int(eval_cfg.inference_delay)
    vis_action_stride = int(eval_cfg.get("vis_action_stride", 4))
    generate_report = bool(eval_cfg.get("generate_html_report", True))
    all_pred = []
    all_gt = []
    vis_data_list = []

    collected = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = build_eval_inputs(batch, device, dtype)

        # Build RTC prefix from GT actions (already in normalized space)
        if inference_delay > 0:
            gt_normalized = inputs["actions"]
            action_horizon = gt_normalized.shape[1]
            prefix = gt_normalized[:, :inference_delay, :]
            if inference_delay < action_horizon:
                pad = prefix.new_zeros(
                    prefix.shape[0], action_horizon - inference_delay, prefix.shape[2],
                )
                prefix = torch.cat([prefix, pad], dim=1)
            inputs["prev_action_chunk"] = prefix
            inputs["inference_delay"] = inference_delay

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            pred_actions = model("infer_action", inputs)

        valid_mask = batch["actions_valid_mask"].bool()
        pred_unnorm = normalizer[norm_key].unnormalize(pred_actions.float().cpu())
        gt_unnorm = normalizer[norm_key].unnormalize(batch["actions"].float())
        all_pred.append(pred_unnorm * valid_mask)
        all_gt.append(gt_unnorm * valid_mask)
        collected += pred_unnorm.shape[0]

        # Collect visualization data for all samples
        if generate_report:
            B = pred_unnorm.shape[0]
            head_video_indices = build_head_video_indices(batch, batch.get("video_grid_thw"))
            frames = recover_frames_from_pixel_values_videos(
                batch.get("pixel_values_videos"),
                batch.get("video_grid_thw"),
                head_video_indices if head_video_indices is not None else list(range(B)),
            )
            for j in range(B):
                vis_sample = prepare_vis_sample(
                    batch=batch, frame=frames[j],
                    pred_actions=pred_unnorm[j], gt_actions=gt_unnorm[j],
                    sample_idx=j, normalizer=normalizer,
                    use_relative_action=use_relative_action,
                    global_index=len(vis_data_list),
                )
                vis_data_list.append(vis_sample)

        if collected >= eval_cfg.num_samples:
            break

    if not all_pred:
        print("ERROR: No samples were evaluated. Check dataset paths and shards.")
        sys.exit(1)

    pred_all = torch.cat(all_pred, dim=0)
    gt_all = torch.cat(all_gt, dim=0)
    total = pred_all.shape[0]
    print(f"Total: {total} samples from {num_shards} shards")

    # ---- 7. Subsample if needed ----
    if total > eval_cfg.num_samples:
        indices = rng.choice(total, size=eval_cfg.num_samples, replace=False)
        indices.sort()
        pred_all = pred_all[indices]
        gt_all = gt_all[indices]
        print(f"Subsampled to {eval_cfg.num_samples} samples")

    # ---- 8. Save full predictions ----
    np.savez_compressed(
        output_dir / "predictions.npz",
        pred_actions=pred_all.numpy(),
        gt_actions=gt_all.numpy(),
    )
    print(f"Saved predictions to {output_dir / 'predictions.npz'}")

    # ---- 9. RTC: sanity check prefix, then exclude from metrics ----
    if inference_delay > 0:
        prefix_pred = pred_all[:, :inference_delay, :]
        prefix_gt = gt_all[:, :inference_delay, :]
        prefix_l1 = torch.mean(torch.abs(prefix_pred - prefix_gt)).item()
        print(f"\nRTC sanity check (inference_delay={inference_delay}):")
        print(f"  Prefix steps L1 (should be ~0): {prefix_l1:.6f}")

        # Exclude prefix steps so metrics only reflect denoised actions
        pred_all = pred_all[:, inference_delay:, :]
        gt_all = gt_all[:, inference_delay:, :]
        print(f"  Metrics computed on steps [{inference_delay}:] only")

    # ---- 10. Compute and save metrics ----
    from src.utils.metric import compute_and_save_metrics_from_data

    compute_and_save_metrics_from_data(
        data={"pred_actions": pred_all, "gt_actions": gt_all},
        output_dir=output_dir,
        dataset_name="eval_dataset",
    )

    # ---- 11. Generate HTML report with 2D action visualization ----
    if generate_report and vis_data_list:
        print("\nGenerating 2D visualizations and HTML report...")
        vis_images = []
        for vd in vis_data_list:
            vis_img = render_overlay_image(vd, action_stride=vis_action_stride)
            vis_images.append((vd["global_index"], vd["sample_mae"], vis_img))

        metric_png_names = [
            "smoothness_analysis.png", "error_heatmap.png",
            "covariance_matrices.png", "loss_over_time.png",
            "trajectory_comparison.png", "smoothness_error_heatmaps.png",
        ]
        metric_pngs = [output_dir / n for n in metric_png_names if (output_dir / n).exists()]

        report_path = generate_html_report(
            output_dir=output_dir,
            eval_config=OmegaConf.to_container(eval_cfg, resolve=True),
            train_config_path=str(eval_cfg.train_config_path),
            checkpoint_path=str(eval_cfg.checkpoint_path),
            metrics_json_path=output_dir / "metrics.json",
            metric_png_paths=metric_pngs,
            vis_images=vis_images,
        )
        print(f"HTML report: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
