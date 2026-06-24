from __future__ import annotations

from typing import Any

import torch



def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _build_action_valid_mask(batch: dict[str, Any], action_horizon: int, action_dim: int) -> torch.Tensor:
    if "actions_valid_mask" in batch:
        return batch["actions_valid_mask"].to(dtype=torch.bool)

    n_actions = batch["n_actions"]
    device = n_actions.device
    step_mask = torch.arange(action_horizon, device=device).unsqueeze(0) < n_actions.unsqueeze(1)
    return step_mask.unsqueeze(-1).expand(-1, -1, action_dim)


def _last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.shape[1]
    positions = torch.arange(seq_len, device=attention_mask.device, dtype=torch.long).unsqueeze(0)
    masked_positions = positions.masked_fill(attention_mask == 0, -1)
    last_indices = masked_positions.max(dim=1).values
    return last_indices.clamp(min=0)


def prepare_prefix_memory(model, batch: dict, output_attentions: bool = False):
    prefix_batch = _clone_batch(batch)
    prefix_batch.pop("actions", None)
    prefix_batch.pop("actions_valid_mask", None)
    slot_embeds = model.build_slot_embeddings(prefix_batch)
    return model.forward_backbone_stream(prefix_batch, slot_embeds, output_attentions=output_attentions)


def infer_flow_action(model, batch: dict, **kwargs):
    prev_action_chunk = batch.pop("prev_action_chunk", None)
    inference_delay = int(batch.pop("inference_delay", 0))
    output_attentions = kwargs.get("output_attentions", False)

    working_batch = _clone_batch(batch)

    action_valid_mask = _build_action_valid_mask(working_batch, model.num_action_tokens, model.action_dim)
    batch_size, action_len, _ = action_valid_mask.shape
    device = working_batch["input_ids"].device
    action_dtype = working_batch["states"].dtype

    generated_actions = torch.randn(batch_size, action_len, model.action_dim, device=device, dtype=action_dtype)
    action_step_mask = action_valid_mask.any(dim=-1)

    # RTC: pin prefix actions to previously executed values.
    use_rtc = prev_action_chunk is not None and inference_delay > 0
    prefix_token_mask = torch.zeros(batch_size, action_len, dtype=torch.bool, device=device)
    if use_rtc:
        prev_action_chunk = prev_action_chunk.to(device=device, dtype=action_dtype)
        assert prev_action_chunk.shape[1] == action_len, (
            f"prev_action_chunk length {prev_action_chunk.shape[1]} != action_horizon {action_len}"
        )
        assert inference_delay <= action_len, (
            f"inference_delay {inference_delay} > action_horizon {action_len}"
        )
        prefix_token_mask[:, :inference_delay] = True

    working_batch["actions_valid_mask"] = action_valid_mask

    backbone_output = prepare_prefix_memory(model, working_batch, output_attentions=output_attentions)
    vlm_attn_weights = backbone_output.attention_weights if output_attentions else None

    delta_t = 1.0 / max(model.num_inference_steps, 1)
    t = torch.zeros(batch_size, device=device, dtype=action_dtype)
    expert_attn_per_step: list | None = [] if output_attentions else None

    for _ in range(model.num_inference_steps):
        if use_rtc:
            generated_actions = torch.where(
                prefix_token_mask.unsqueeze(-1),
                prev_action_chunk,
                generated_actions,
            )
            time_for_model = torch.where(
                prefix_token_mask,
                torch.ones(batch_size, action_len, device=device, dtype=action_dtype),
                t[:, None].expand(-1, action_len),
            )
        else:
            time_for_model = t[:, None].expand(-1, action_len)

        flow_inputs = {
            "noisy_actions": generated_actions,
            "time_for_model": time_for_model,
        }
        flow_output = model.forward_flow_stream(
            batch=working_batch,
            backbone_output=backbone_output,
            flow_inputs=flow_inputs,
            num_parallel_chunks=1,
            output_attentions=output_attentions,
        )
        generated_actions = generated_actions + delta_t * flow_output["pred_v"]
        t = t + delta_t
        if output_attentions:
            expert_attn_per_step.append(flow_output["expert_attention_weights"])

    if use_rtc:
        generated_actions = torch.where(
            prefix_token_mask.unsqueeze(-1),
            prev_action_chunk,
            generated_actions,
        )

    result = generated_actions * action_step_mask.unsqueeze(-1).to(dtype=generated_actions.dtype)
    if output_attentions:
        return {
            "generated_actions": result,
            "vlm_attention_weights": vlm_attn_weights,
            "expert_attention_weights": expert_attn_per_step,
        }
    return result
