'''
Evaluation and checkpoint utilities for EgoSteer training workspace.

Extracted from train_egosteer_workspace.py for modularity.
'''

import gc
import os
import shutil
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from contextlib import contextmanager
import pathlib

from src.utils.metric import get_action_accuracy
from src.utils.fsdp_app_state import APP_STATE_KEY, FSDPWorkspaceAppState


def clear_attn_weights(model):
    if hasattr(model, "joint_model") and hasattr(model.joint_model, "attn_weights"):
        model.joint_model.attn_weights = [None] * model.joint_model.num_hidden_layers


def _unwrap_model(workspace):
    if hasattr(workspace.model, 'module'):
        return workspace.model.module
    return workspace.model


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


@contextmanager
def eval_with_unsharded_model(model):
    """Set up model for FSDP2-safe evaluation.

    Explicitly unshard all FSDP modules before the eval loop so that every
    forward pass is purely local — no per-forward collective all-gather.
    Combined with ``reshard_after_forward=False`` this means ranks with
    unequal or even zero validation batches will NOT deadlock.  The unshard
    itself is the only collective; all ranks participate because it runs
    before the iteration loop.
    """
    from src.utils.distributed_utils import _collect_fsdp_modules

    fsdp_modules = _collect_fsdp_modules(model)
    for m in fsdp_modules:
        m.unshard()

    model.eval()
    try:
        yield
    finally:
        model.train()
        for m in fsdp_modules:
            m.reshard()


def save_checkpoint_native(workspace, rank, path=None, tag='latest'):
    if path is None:
        path = pathlib.Path(workspace.output_dir).joinpath('checkpoints', f'{tag}')
    else:
        path = pathlib.Path(path)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[ckpt] saving update_step={workspace.update_step} "
            f"epoch={workspace.epoch} -> {path} "
            f"(model + optimizer + lr_scheduler + training_state)"
        )
    dist.barrier()
    workspace.training_state.update_step = workspace.update_step
    workspace.training_state.global_step = workspace.global_step
    workspace.training_state.epoch = workspace.epoch
    app_state = FSDPWorkspaceAppState(
        model=workspace.model,
        optimizer=workspace.optimizer,
        lr_scheduler=workspace.lr_scheduler,
        training_state=workspace.training_state,
    )
    dcp.save({APP_STATE_KEY: app_state}, checkpoint_id=str(path))
    if rank == 0:
        print(f"[ckpt] saved -> {path}")


def save_topk_ckpt(workspace, rank, topk_manager, step_log):
    # Need to update_bn when the model contains batch norm layers !!!
    if workspace.cfg.checkpoint.save_last_ckpt:
        save_checkpoint_native(workspace, rank)

    # sanitize metric names
    metric_dict = dict()
    for key, value in step_log.items():
        new_key = key.replace('/', '_')
        metric_dict[new_key] = value

    # Atomic top-k swap: propose -> save -> verify .metadata -> commit.
    # The displaced ckpt is deleted only after the new save is verified durable,
    # so any failure (exception, missing .metadata, partial write) leaves the
    # existing top-k intact and path_value_map free of ghost entries.
    new_path, delete_path, value = topk_manager.propose_ckpt_path(metric_dict)
    if new_path is None:
        return

    save_failed = False
    try:
        save_checkpoint_native(workspace, rank, path=new_path)
    except Exception:
        save_failed = True
        raise
    finally:
        if save_failed and rank == 0 and os.path.exists(new_path):
            if os.path.isdir(new_path):
                shutil.rmtree(new_path, ignore_errors=True)
            else:
                try:
                    os.remove(new_path)
                except OSError:
                    pass

    # Sync all ranks finished writing, then verify DCP's commit marker.
    dist.barrier()
    metadata_file = os.path.join(new_path, ".metadata")
    if not os.path.exists(metadata_file):
        if rank == 0 and os.path.isdir(new_path):
            shutil.rmtree(new_path, ignore_errors=True)
        raise RuntimeError(
            f"DCP save returned without error but {metadata_file} is missing; "
            f"top-k ckpt at {new_path} is incomplete. Existing top-k preserved."
        )

    topk_manager.commit(rank, new_path, value, delete_path)


def save_interval_ckpt(workspace, rank):
    save_dir = os.path.join(workspace.output_dir, 'step_checkpoints')
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()
    # Need to update_bn when the model contains batch norm layers !!!
    save_checkpoint_native(workspace, rank, path=os.path.join(save_dir, f'update_step_{workspace.update_step}'))


# ---------------------------------------------------------------------------
# Evaluation sub-routines
# ---------------------------------------------------------------------------

def compute_batch_action_metrics(
    workspace, inputs, eval_thresholds, wrist_trans_dim, wrist_dim,
):
    """Compute action accuracy and L1 metrics for a single eval batch.

    Returns None if no valid actions, otherwise a dict with keys:
        accuracy, l1_loss, l1_parts, per_sample_l1, eval_sample
    """
    gt_actions = inputs['actions']
    actions_valid_mask = inputs['actions_valid_mask']
    if not torch.any(actions_valid_mask):
        return None

    with torch.amp.autocast("cuda", dtype=workspace.dtype):
        pred_actions = workspace.model("infer_action", inputs)

    B, H, D = gt_actions.shape
    eval_sample = torch.any(actions_valid_mask.reshape(B, -1), dim=1)
    actions_valid_mask = actions_valid_mask[eval_sample]

    # Unnormalize to original scale
    if workspace.use_relative_action:
        gt_actions = workspace.normalizer['actions'].unnormalize(gt_actions[eval_sample])
        pred_actions = workspace.normalizer['actions'].unnormalize(pred_actions[eval_sample])
    else:
        gt_actions = workspace.normalizer['motions'].unnormalize(gt_actions[eval_sample])
        pred_actions = workspace.normalizer['motions'].unnormalize(pred_actions[eval_sample])
    gt_actions = gt_actions * actions_valid_mask
    pred_actions = pred_actions * actions_valid_mask

    accuracy = get_action_accuracy(gt_actions, pred_actions, eval_thresholds, valid_mask=actions_valid_mask)

    abs_diff = torch.abs(pred_actions - gt_actions)
    per_sample_l1 = (
        torch.sum(abs_diff.flatten(start_dim=1), dim=1)
        / torch.sum(actions_valid_mask.flatten(start_dim=1), dim=1)
    )
    l1_loss = torch.sum(abs_diff) / torch.sum(actions_valid_mask)

    # Per-part L1 loss
    l1_parts = {}
    for pname, ps, pe in [
        ("wrist_trans", 0, wrist_trans_dim),
        ("wrist_rot", wrist_trans_dim, wrist_dim),
        ("hand", wrist_dim, D),
    ]:
        pvalid = actions_valid_mask[:, :, ps:pe].sum().clamp(min=1)
        l1_parts[pname] = torch.sum(abs_diff[:, :, ps:pe]) / pvalid

    return {
        "accuracy": accuracy,
        "l1_loss": l1_loss,
        "l1_parts": l1_parts,
        "per_sample_l1": per_sample_l1,
        "eval_sample": eval_sample,
    }


def update_attn_sample_tracker(
    inputs, full_seq_attn_maps, eval_sample, per_sample_l1,
    batch_idx, min_loss_sample, max_loss_sample,
):
    """Update min/max loss sample trackers for attention visualization."""
    if full_seq_attn_maps is None:
        return

    attn_weights = full_seq_attn_maps[:, eval_sample, :, :, :]
    eval_indices = torch.nonzero(eval_sample, as_tuple=False).squeeze(1)
    batch_size = inputs["input_ids"].shape[0]

    def build_sample_inputs(sample_batch_idx):
        sample_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.shape[0] == batch_size:
                sample_inputs[key] = _to_numpy(value[sample_batch_idx])
            else:
                sample_inputs[key] = _to_numpy(value)
        return sample_inputs

    min_idx = torch.argmin(per_sample_l1).item()
    max_idx = torch.argmax(per_sample_l1).item()
    min_loss = per_sample_l1[min_idx].item()
    max_loss = per_sample_l1[max_idx].item()
    min_batch_idx = eval_indices[min_idx].item()
    max_batch_idx = eval_indices[max_idx].item()

    if min_loss < min_loss_sample['loss']:
        min_loss_sample['loss'] = min_loss
        min_loss_sample['attn_weights'] = attn_weights[:, min_idx, :, :, :].float().cpu()
        min_loss_sample['inputs'] = build_sample_inputs(min_batch_idx)
        min_loss_sample['metadata'] = {
            'batch_idx': batch_idx, 'sample_idx': min_batch_idx,
            'eval_sample_idx': min_idx, 'l1_loss': min_loss,
        }

    if max_loss > max_loss_sample['loss']:
        max_loss_sample['loss'] = max_loss
        max_loss_sample['attn_weights'] = attn_weights[:, max_idx, :, :, :].float().cpu()
        max_loss_sample['inputs'] = build_sample_inputs(max_batch_idx)
        max_loss_sample['metadata'] = {
            'batch_idx': batch_idx, 'sample_idx': max_batch_idx,
            'eval_sample_idx': max_idx, 'l1_loss': max_loss,
        }


def aggregate_val_losses(val_losses, device, step_log):
    """Reduce per-batch validation losses across all processes and write to step_log."""
    for key in val_losses.keys():
        if len(val_losses[key]) == 0:
            local_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
            num_nonzero_samples = torch.tensor(0, device=device)
        else:
            stacked_losses = torch.stack(val_losses[key])
            # Count non-zero losses (losses > 1e-8 to handle floating point precision)
            nonzero_mask = stacked_losses > 1e-8
            num_nonzero_samples = nonzero_mask.sum().to(device)
            local_loss_sum = stacked_losses.sum().to(device)

        dist.all_reduce(num_nonzero_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)

        # Average only over non-zero samples
        val_losses[key] = (local_loss_sum / num_nonzero_samples.clamp(min=1)).item()
        step_log[f'val_loss/{key}'] = val_losses[key]


def aggregate_action_metrics(
    eval_accuracy, eval_l1_loss, eval_l1_loss_parts,
    eval_thresholds, device, step_log,
):
    """Reduce action metrics across processes, write to step_log, and return averaged values."""
    eval_len = len(eval_accuracy)

    if eval_len > 0:
        sum_accuracy = torch.stack(eval_accuracy).sum(dim=0).to(device)
        sum_l1 = torch.stack(eval_l1_loss).sum().to(device)
        sum_l1_parts = {k: torch.stack(v).sum().to(device) for k, v in eval_l1_loss_parts.items()}
    else:
        sum_accuracy = torch.zeros(len(eval_thresholds), device=device)
        sum_l1 = torch.tensor(0.0, device=device)
        sum_l1_parts = {k: torch.tensor(0.0, device=device) for k in eval_l1_loss_parts}

    count = torch.tensor(eval_len, device=device)
    dist.all_reduce(sum_accuracy, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_l1, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    avg_accuracy = sum_accuracy / count.clamp(min=1)
    avg_l1 = sum_l1 / count.clamp(min=1)
    avg_l1_parts = {}
    for k, v in sum_l1_parts.items():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        avg_l1_parts[k] = v / count.clamp(min=1)

    step_log['eval/l1_loss'] = avg_l1.item()
    for part_name, part_loss in avg_l1_parts.items():
        step_log[f'eval/l1_{part_name}'] = part_loss.item()
    for i, threshold in enumerate(eval_thresholds):
        step_log[f'eval/acc_{threshold}'] = avg_accuracy[i].item()

    return avg_l1, avg_l1_parts, avg_accuracy


def save_attn_samples(workspace, rank, min_loss_sample, max_loss_sample):
    """Save attention weights for min/max loss samples to disk (main process only)."""
    if rank != 0 or min_loss_sample['attn_weights'] is None:
        return

    print(f"\nSaving attention weights and inputs for selected samples...")
    selected_samples = {
        'lowest_loss': min_loss_sample,
        'highest_loss': max_loss_sample,
    }

    print(f"Selected samples for visualization:")
    print(f"  Lowest loss: batch {min_loss_sample['metadata']['batch_idx']}, "
          f"sample {min_loss_sample['metadata']['sample_idx']}, loss={min_loss_sample['loss']:.4f}")
    print(f"  Highest loss: batch {max_loss_sample['metadata']['batch_idx']}, "
          f"sample {max_loss_sample['metadata']['sample_idx']}, loss={max_loss_sample['loss']:.4f}")

    for name, sample_data in selected_samples.items():
        if sample_data['attn_weights'] is None:
            continue

        print(f"\nProcessing {name} sample...")

        output_dir = os.path.join(
            workspace.output_dir,
            'attention_visualization',
            f'step_{workspace.update_step}',
            name,
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'attention_and_inputs.npz')
        save_payload = dict(sample_data['inputs'])
        save_payload['attn_weights'] = sample_data['attn_weights'].numpy()
        save_payload['metadata'] = np.array(sample_data['metadata'], dtype=object)
        save_payload['update_step'] = np.array(workspace.update_step)
        try:
            np.savez_compressed(output_path, **save_payload)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  Error saving attention data: {e}")


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluation(workspace, rank, device, dataloader, step_log):
    if rank == 0:
        print(f"Evaluation step {workspace.update_step} started")
    dist.barrier()

    model = _unwrap_model(workspace)
    wrist_dim = model.shape_meta["obs"]["state"]["wrist"]["shape"][0]
    wrist_trans_dim = 6  # 2 wrists * 3 xyz, fixed layout

    with (
        torch.compiler.set_stance("force_eager"),
        torch.no_grad(),
        eval_with_unsharded_model(workspace.model),
    ):
        val_losses = {"total_loss": [], "ce_loss": [], "flow_loss": [], "wm_loss": []}
        eval_thresholds = workspace.cfg.training.eval_thresholds
        eval_accuracy = []
        eval_l1_loss = []
        eval_l1_loss_parts = {"wrist_trans": [], "wrist_rot": [], "hand": []}

        min_loss_sample = {'loss': float('inf'), 'attn_weights': None, 'inputs': None, 'metadata': None}
        max_loss_sample = {'loss': float('-inf'), 'attn_weights': None, 'inputs': None, 'metadata': None}
        save_eval_attn_weights = bool(workspace.cfg.training.save_eval_attn_weights)

        for batch_idx, batch in enumerate(dataloader):
            # cast_forward_inputs=True on the FSDP MixedPrecisionPolicy casts
            # floating-point batch tensors at model.forward entry, so no
            # manual preprocess is needed here.
            inputs = batch

            # Compute validation loss
            with torch.amp.autocast("cuda", dtype=workspace.dtype), torch.inference_mode():
                loss = workspace.model(
                    "train", inputs,
                    return_attn_weights=save_eval_attn_weights,
                )
            for key, loss_ in loss.items():
                val_losses[key].append(loss_.detach())

            # Attention maps for visualization
            full_seq_attn_maps = None
            if save_eval_attn_weights and hasattr(model, 'attn_weights') and len(model.attn_weights) > 0:
                full_seq_attn_maps = torch.stack(model.attn_weights, dim=0)

            # Action metrics
            if 'actions' in inputs:
                metrics = compute_batch_action_metrics(
                    workspace, inputs,
                    eval_thresholds, wrist_trans_dim, wrist_dim,
                )
                if metrics is None:
                    continue
                eval_accuracy.append(metrics["accuracy"])
                eval_l1_loss.append(metrics["l1_loss"])
                for k, v in metrics["l1_parts"].items():
                    eval_l1_loss_parts[k].append(v)

                update_attn_sample_tracker(
                    inputs, full_seq_attn_maps, metrics["eval_sample"],
                    metrics["per_sample_l1"], batch_idx,
                    min_loss_sample, max_loss_sample,
                )

            if workspace.cfg.training.max_eval_steps and batch_idx >= (workspace.cfg.training.max_eval_steps - 1):
                break

        # Aggregate across processes
        aggregate_val_losses(val_losses, device, step_log)
        avg_l1, avg_l1_parts, avg_accuracy = aggregate_action_metrics(
            eval_accuracy, eval_l1_loss, eval_l1_loss_parts,
            eval_thresholds, device, step_log,
        )

        # Print summary
        log_msg = f"Eval | Epoch {workspace.epoch} | L1 Loss: {avg_l1.item():.3f} | "
        log_msg += " | ".join([f"{k}: {v.item():.3f}" for k, v in avg_l1_parts.items()])
        log_msg += " | "
        log_msg += " | ".join([
            f"acc thres {threshold}: {avg_accuracy[i].item():.3f}"
            for i, threshold in enumerate(eval_thresholds)
        ])
        if rank == 0:
            print(log_msg)

        save_attn_samples(workspace, rank, min_loss_sample, max_loss_sample)

    clear_attn_weights(model)
    # FSDP2's post_forward_order (PyTorch source comment: "will cause ref
    # cycles") holds pinned tensor wrappers alive across eval; a gen-2 gc
    # collect breaks those cycles so pinned storage can return to
    # CachingHostAllocator's free list instead of accumulating in the pool.
    # Ref: https://github.com/pytorch/pytorch/issues/97432
    gc.collect(generation=2)
