"""
Loss computation functions extracted from EgoSteer.

Contains:
- compute_celoss: Cross-entropy loss for language modeling
- compute_loss: Combined VLA loss (CE + flow + world model)
"""

import torch
from torch import nn

from src.utils.sample_utils import sample_flow_time, sample_rtc_delay


def compute_celoss(
    lm_head: nn.Module,
    ce_loss_fn: nn.Module,
    ignore_index: int,
    hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
) -> torch.FloatTensor:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        lm_head: Language model head module
        ce_loss_fn: Cross-entropy loss function (sum reduction)
        ignore_index: Label index to ignore in loss computation
        hidden_states: [B, seq_len, hidden_size] Hidden states from the language model
        labels: [B, seq_len] Labels for language modeling loss

    Returns:
        torch.FloatTensor: Normalized cross-entropy loss
    """
    logits = lm_head(hidden_states)
    logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
    labels = labels[:, 1:].contiguous().view(-1)
    ce_loss = ce_loss_fn(logits, labels)
    valid_num_labels = torch.sum(labels != ignore_index)
    return ce_loss / valid_num_labels.clamp(min=1)


def psi_t(
    x: torch.FloatTensor,
    x1: torch.FloatTensor,
    t: torch.FloatTensor,
    flow_sig_min: float,
) -> torch.FloatTensor:
    """
    Conditional flow function for flow matching.

    Interpolates between noise x and target x1 based on time t.

    Args:
        x: [B, horizon_steps, action_dim] Initial noise
        x1: [B, horizon_steps, action_dim] Target action
        t: [B] Time parameter (0 to 1)
        flow_sig_min: Minimum sigma for flow matching

    Returns:
        torch.FloatTensor: [B, horizon_steps, action_dim] Interpolated action at time t
    """
    if t.ndim == 1:
        t = t[:, None, None]
    elif t.ndim == 2:
        t = t[:, :, None]
    else:
        raise ValueError(f"Unsupported time shape for psi_t: {t.shape}")

    return (1 - (1 - flow_sig_min) * t) * x + t * x1


def build_rtc_flow_inputs(
    *,
    actions: torch.FloatTensor,
    actions_valid_mask: torch.Tensor,
    postfix_time: torch.FloatTensor,
    n_actions: torch.LongTensor | None,
    rtc_delay_strategy: str = "uniform",
    rtc_max_delay: int | None = None,
    forced_delay: torch.LongTensor | None = None,
) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.FloatTensor, torch.LongTensor]:
    """
    Build token-wise RTC conditioning inputs for flow-matching training.

    This helper converts a per-sample postfix diffusion time into token-wise
    times by assigning:
    - time = 1.0 for prefix tokens that are treated as known action conditions
    - time = postfix_time for postfix tokens that remain denoising targets

    It also constructs the postfix validity mask used to restrict the flow loss
    to valid non-prefix action tokens.

    Args:
        actions:
            [B, H, D] Action targets, where B is batch size, H is action horizon,
            and D is action dimension.
        actions_valid_mask:
            [B, H, D] or broadcast-compatible mask indicating which action values
            are valid for loss computation. Padding positions must be False.
        postfix_time:
            [B] Per-sample flow time used for RTC postfix tokens.
        n_actions:
            Optional [B] tensor containing the number of valid action steps for
            each sample. If None, valid lengths are inferred from
            actions_valid_mask.
        rtc_max_delay:
            Optional cap on the sampled prefix length.
        forced_delay:
            Optional [B] deterministic delay tensor for tests/debugging.

    Returns:
        tuple containing:
            - token_t (torch.FloatTensor):
              [B, H] Token-wise flow times, equal to 1.0 on the prefix and
              postfix_time on the postfix.
            - postfix_valid_mask (torch.BoolTensor):
              [B, H, D]-compatible boolean mask that is True only on postfix
              positions that are also valid action targets.
            - prefix_mask (torch.BoolTensor):
              [B, H] Boolean mask marking the RTC action prefix.
            - delay (torch.LongTensor):
              [B] Sampled prefix lengths.
    """
    batch_size, horizon_steps = actions.shape[:2]
    device = actions.device

    if n_actions is not None:
        valid_action_len = n_actions.to(device=device, dtype=torch.long)
    else:
        valid_action_len = actions_valid_mask.reshape(batch_size, horizon_steps, -1).any(dim=-1).sum(dim=1)
    valid_action_len = valid_action_len.clamp(min=0, max=horizon_steps)

    delay = sample_rtc_delay(
        valid_action_len,
        strategy=rtc_delay_strategy,
        max_delay=rtc_max_delay,
        forced_delay=forced_delay,
    )
    positions = torch.arange(horizon_steps, device=device).unsqueeze(0)
    prefix_mask = positions < delay.unsqueeze(1)
    token_t = torch.where(prefix_mask, torch.ones_like(postfix_time[:, None]), postfix_time[:, None])
    postfix_mask = (~prefix_mask).unsqueeze(-1)
    postfix_valid_mask = postfix_mask & actions_valid_mask.to(dtype=torch.bool)
    return token_t, postfix_valid_mask, prefix_mask, delay

def compute_packed_flow_loss(
    *,
    model,
    actions: torch.FloatTensor,
    pred_v_t: torch.FloatTensor,
    noise: torch.FloatTensor,
    loss_mask: torch.Tensor,
) -> torch.FloatTensor:
    """Compute masked flow-matching regression loss with global normalization.

    All inputs are pre-packed by build_flow_inputs to shape [B, T*H, D],
    so this function is a simple element-wise masked MSE.

    Args:
        model: EgoSteer model instance (uses model.flow_config.sig_min).
        actions: [B, T*H, D] Packed ground-truth actions (repeated per chunk).
        pred_v_t: [B, T*H, D] Packed predicted flow velocity.
        noise: [B, T*H, D] Packed Gaussian noise.
        loss_mask: [B, T*H, D] Boolean mask for valid loss positions.
    """
    target_v = actions - (1 - model.flow_config.sig_min) * noise
    squared_error = (pred_v_t - target_v) ** 2
    mask_float = loss_mask.to(dtype=squared_error.dtype)
    return (squared_error * mask_float).sum() / mask_float.sum().clamp(min=1)

def zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def compute_ce_loss(model, hidden_states: torch.Tensor, labels: torch.Tensor, is_vla_data: torch.Tensor) -> torch.Tensor:
    non_vla_mask = ~is_vla_data.to(dtype=torch.bool)
    if not torch.any(non_vla_mask):
        return zero_loss(hidden_states)
    return compute_celoss(
        model.lm_head,
        model.CELoss,
        model.ignore_index,
        hidden_states[non_vla_mask],
        labels[non_vla_mask],
    )


def build_flow_inputs(
    model,
    batch: dict[str, torch.Tensor | None],
    *,
    num_parallel_t: int | None = None,
    sampled_t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    actions = batch["actions"]
    actions_valid_mask = batch["actions_valid_mask"]
    n_actions = batch.get("n_actions")
    batch_size, horizon_steps, action_dim = actions.shape
    if num_parallel_t is None:
        num_parallel_t = model.flow_config.num_parallel_t
    if num_parallel_t < 1:
        raise ValueError(f"num_parallel_t must be >= 1, got {num_parallel_t}.")

    if sampled_t is None:
        sampled_t = sample_flow_time(
            batch_size, num_parallel_t,
            sampling=model.flow_config.sampling,
            alpha=model.flow_config.alpha,
            beta=model.flow_config.beta,
            sig_min=model.flow_config.sig_min,
        )
    sampled_t = sampled_t.to(device=actions.device, dtype=actions.dtype)
    if sampled_t.ndim == 1:
        if num_parallel_t != 1 or sampled_t.shape[0] != batch_size:
            raise ValueError(
                f"Expected sampled flow time to have shape [{batch_size}] for num_parallel_t=1, got {sampled_t.shape}."
            )
        sampled_t = sampled_t.unsqueeze(1)
    elif sampled_t.shape != (batch_size, num_parallel_t):
        raise ValueError(
            f"Expected sampled flow time to have shape [{batch_size}, {num_parallel_t}], got {sampled_t.shape}."
        )

    expanded_actions = actions.unsqueeze(1).expand(-1, num_parallel_t, -1, -1)
    flat_actions = expanded_actions.reshape(batch_size * num_parallel_t, horizon_steps, action_dim)
    flat_noise = torch.randn_like(flat_actions)

    prefix_mask = None
    rtc_mask = None
    if model.rtc_config.enabled:
        expanded_valid_mask = actions_valid_mask.unsqueeze(1).expand(-1, num_parallel_t, -1, -1)
        flat_n_actions = None
        if n_actions is not None:
            flat_n_actions = n_actions.unsqueeze(1).expand(-1, num_parallel_t).reshape(-1)
        flat_time_for_model, flat_rtc_mask, flat_prefix_mask, _ = build_rtc_flow_inputs(
            actions=flat_actions,
            actions_valid_mask=expanded_valid_mask.reshape(batch_size * num_parallel_t, horizon_steps, action_dim),
            postfix_time=sampled_t.reshape(-1),
            n_actions=flat_n_actions,
            rtc_delay_strategy=model.rtc_config.delay_strategy,
            rtc_max_delay=model.rtc_config.max_delay,
        )
        time_for_model = flat_time_for_model.reshape(batch_size, num_parallel_t, horizon_steps)
        rtc_mask = flat_rtc_mask.reshape(batch_size, num_parallel_t, horizon_steps, action_dim)
        prefix_mask = flat_prefix_mask.reshape(batch_size, num_parallel_t, horizon_steps)
    else:
        time_for_model = sampled_t.unsqueeze(-1).expand(-1, -1, horizon_steps)

    flat_noisy_actions = psi_t(
        flat_noise,
        flat_actions,
        time_for_model.reshape(batch_size * num_parallel_t, horizon_steps),
        model.flow_config.sig_min,
    )
    if prefix_mask is not None:
        flat_noisy_actions = torch.where(
            prefix_mask.reshape(batch_size * num_parallel_t, horizon_steps).unsqueeze(-1),
            flat_actions,
            flat_noisy_actions,
        )

    # Build packed loss mask: [B, T*H, D]
    if rtc_mask is not None:
        loss_mask = rtc_mask.reshape(batch_size, num_parallel_t * horizon_steps, action_dim)
    else:
        loss_mask = actions_valid_mask.unsqueeze(1).expand(-1, num_parallel_t, -1, -1).reshape(
            batch_size, num_parallel_t * horizon_steps, action_dim,
        )

    return {
        "noisy_actions": flat_noisy_actions.reshape(batch_size, num_parallel_t * horizon_steps, action_dim),
        "time_for_model": time_for_model.reshape(batch_size, num_parallel_t * horizon_steps),
        "noise": flat_noise.reshape(batch_size, num_parallel_t * horizon_steps, action_dim),
        "actions": flat_actions.reshape(batch_size, num_parallel_t * horizon_steps, action_dim),
        "loss_mask": loss_mask,
    }


def compute_flow_stream_loss(
    model,
    batch: dict[str, torch.Tensor | None],
    backbone_output,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    if "actions" not in batch or "actions_valid_mask" not in batch:
        return zero_loss(backbone_output.last_hidden_states), {}

    flow_batch = dict(batch)
    if "is_vla_data" in batch:
        is_vla_data = batch["is_vla_data"].to(device=batch["actions_valid_mask"].device, dtype=torch.bool)
        flow_batch["actions_valid_mask"] = batch["actions_valid_mask"] & is_vla_data[:, None, None]
        if "n_actions" in batch:
            flow_batch["n_actions"] = torch.where(
                is_vla_data,
                batch["n_actions"],
                torch.zeros_like(batch["n_actions"]),
            )
        if not torch.any(flow_batch["actions_valid_mask"]):
            return zero_loss(backbone_output.last_hidden_states), {}

    num_parallel_t = model.flow_config.num_parallel_t
    flow_inputs = build_flow_inputs(model, flow_batch, num_parallel_t=num_parallel_t)
    flow_output = model.forward_flow_stream(
        batch=flow_batch,
        backbone_output=backbone_output,
        flow_inputs=flow_inputs,
        num_parallel_chunks=num_parallel_t,
    )
    flow_loss = compute_packed_flow_loss(
        model=model,
        actions=flow_inputs["actions"],
        pred_v_t=flow_output["pred_v"],
        noise=flow_inputs["noise"],
        loss_mask=flow_inputs["loss_mask"],
    )
    flow_output["flow_loss"] = flow_loss
    return flow_loss, flow_output


def compute_wm_loss(
    model,
    batch: dict[str, torch.Tensor],
    backbone_output,
) -> torch.Tensor:
    """Masked MSE between world model predictions and frozen teacher features.

    Both pred and target are ``[B, 2, K, spatial, D]`` in canonical
    head/chest order. By default ``view_mask`` controls which views contribute
    loss; ``world_model_config.mask_loss_by_view_mask=False`` supervises all
    views.
    """
    if not model.use_world_model or "future_frames" not in batch:
        return zero_loss(backbone_output.last_hidden_states)

    wm_output = model.forward_world_model_stream(batch, backbone_output)
    pred = wm_output["pred"]
    target = wm_output["target"].to(pred.dtype)
    n_future = wm_output["n_future_frames"].to(device=pred.device)
    V, K = pred.shape[1], pred.shape[2]

    if model.world_model_config.mask_loss_by_view_mask:
        view_mask = wm_output["view_mask"].to(device=pred.device, dtype=torch.bool)
        if view_mask.shape != (pred.shape[0], V):
            raise ValueError(f"view_mask must have shape [B, V], got {tuple(view_mask.shape)}")
    else:
        # force to calculate loss on all views
        view_mask = torch.ones(pred.shape[0], V, device=pred.device, dtype=torch.bool)

    # frame_valid[b, k] == (k < n_future[b]); broadcast over V/spatial/D for
    # element-wise mask against pred [B, V, K, spatial, D].
    # [B, K] + [B, V] -> [B, V, K, 1, 1]
    frame_valid = torch.arange(K, device=n_future.device) < n_future.unsqueeze(1)
    view_frame_valid = frame_valid[:, None, :] & view_mask[:, :, None]
    mask_float = view_frame_valid[:, :, :, None, None].to(dtype=pred.dtype)

    squared_error = (pred - target) ** 2
    n_elements = pred.shape[-2] * pred.shape[-1]
    denom = mask_float.sum().clamp(min=1)
    return (squared_error * mask_float).sum() / denom / n_elements


def compute_total_loss(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    slot_embeds = model.build_slot_embeddings(batch)
    backbone_output = model.forward_backbone_stream(batch, slot_embeds)
    hidden_states = backbone_output.last_hidden_states
    is_vla_data = batch["is_vla_data"].to(dtype=torch.bool)

    ce_loss = compute_ce_loss(model, hidden_states, batch["labels"], is_vla_data)

    flow_loss, _ = compute_flow_stream_loss(model, batch, backbone_output)
    wm_loss = compute_wm_loss(model, batch, backbone_output)

    total_loss = (
        model.loss_config.ce_loss_weight * ce_loss
        + model.loss_config.flow_loss_weight * flow_loss
        + model.loss_config.wm_loss_weight * wm_loss
    )
    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
        "flow_loss": flow_loss,
        "wm_loss": wm_loss,
    }


def compute_loss(model, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
    del kwargs
    return compute_total_loss(model, batch)
