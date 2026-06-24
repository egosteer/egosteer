"""Qwen3-based expert with optional DiT-style AdaLN-Zero modulation.

This module keeps the Qwen3 text attention / MLP / rotary embedding stack
structurally aligned with the backbone LLM. When ``use_adaln=True`` (default),
the two per-layer pre-norms are replaced with AdaLN-Zero conditioning (DiT mode
for flow matching). When ``use_adaln=False``, standard RMSNorm pre-norms are
used (for world model or other non-conditioned experts).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.common.modules import AdaLNZero
from src.model.vlm.prefix_cache import PrefixKVCache

try:
    from transformers import Qwen3VLConfig, Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextAttention,
        Qwen3VLTextMLP,
        Qwen3VLTextRMSNorm,
        Qwen3VLTextRotaryEmbedding,
    )
except ImportError:
    Qwen3VLConfig = None  # type: ignore[assignment,misc]


class StaticPrefixCache:
    """Compile-friendly single-layer KV cache for action expert.

    Mimics the subset of HF DynamicCache interface used by
    Qwen3VLTextAttention.forward (only ``update`` is called when
    past_key_values is not None). Stores a fixed prefix and prepends it
    to incoming key/value states — pure tensor ops, no graph breaks.

    # Source: transformers Qwen3VLTextAttention.forward calls
    #   past_key_values.update(key_states, value_states, self.layer_idx)
    """

    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        # key/value: [B, num_kv_heads, prefix_len, head_dim]
        self.key = key
        self.value = value

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = torch.cat([self.key.to(key_states.dtype), key_states], dim=2)
        value_states = torch.cat([self.value.to(value_states.dtype), value_states], dim=2)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.key.shape[2]


class DiTQwen3DecoderLayer(nn.Module):
    """Qwen3 decoder block with optional AdaLN-Zero conditioning.

    When ``use_adaln=True`` (DiT mode), the two pre-norms are replaced with
    AdaLN-Zero modulation conditioned on an external signal (e.g. flow time).
    When ``use_adaln=False`` (standard mode), plain RMSNorm pre-norms are used
    and no external conditioning is needed.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        attention_cls: type[nn.Module],
        mlp_cls: type[nn.Module],
        use_adaln: bool = True,
        cond_hidden_size: int = 1024,
        use_kv_projection: bool = False,
    ):
        super().__init__()
        self.use_adaln = use_adaln
        self.self_attn = attention_cls(config=config, layer_idx=layer_idx)
        self.mlp = mlp_cls(config)
        if use_adaln:
            self.attn_adaln = AdaLNZero(config.hidden_size, cond_hidden_size, eps=config.rms_norm_eps)
            self.mlp_adaln = AdaLNZero(config.hidden_size, cond_hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if use_kv_projection:
            self.prefix_key_proj = nn.Linear(head_dim, head_dim, bias=False)
            self.prefix_value_proj = nn.Linear(head_dim, head_dim, bias=False)
            nn.init.eye_(self.prefix_key_proj.weight)
            nn.init.eye_(self.prefix_value_proj.weight)
        else:
            self.prefix_key_proj = nn.Identity()
            self.prefix_value_proj = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | Any,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        cond: torch.Tensor | None = None,
        text_position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prefix_key = self.prefix_key_proj(prefix_key)
        prefix_value = self.prefix_value_proj(prefix_value)
        prefix_cache = StaticPrefixCache(prefix_key, prefix_value)

        # Attention block
        residual = hidden_states
        if self.use_adaln:
            attn_inputs, gate = self.attn_adaln(hidden_states, cond)
        else:
            attn_inputs = self.input_layernorm(hidden_states)
            gate = None
        # Second return is attn_weights (eager), LSE (flex), or None (sdpa).
        attn_output, attn_weights = self.self_attn(
            hidden_states=attn_inputs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=prefix_cache,
            use_cache=False,
            is_causal=False,
        )
        hidden_states = residual + (gate * attn_output if gate is not None else attn_output)

        # MLP block
        residual = hidden_states
        if self.use_adaln:
            mlp_inputs, gate = self.mlp_adaln(hidden_states, cond)
        else:
            mlp_inputs = self.post_attention_layernorm(hidden_states)
            gate = None
        mlp_output = self.mlp(mlp_inputs)
        hidden_states = residual + (gate * mlp_output if gate is not None else mlp_output)
        return hidden_states, attn_weights


def compute_kv_layer_indices(num_expert_layers: int, num_backbone_layers: int) -> list[int]:
    """Compute which backbone KV layers to use when expert has fewer layers.

    Uniformly samples ``num_expert_layers`` indices from [0, num_backbone_layers).
    When counts match, returns identity mapping.
    """
    if num_expert_layers >= num_backbone_layers:
        return list(range(num_backbone_layers))
    step = num_backbone_layers / num_expert_layers
    return [int(step * i) for i in range(num_expert_layers)]


class Qwen3Expert(nn.Module):
    """Qwen3-based expert reusing text attention / MLP / rotary primitives.

    Supports two modes via ``use_adaln``:
    - ``True`` (DiT mode): AdaLN-Zero conditioning for flow matching action expert.
    - ``False`` (standard mode): RMSNorm pre-norms for world model or other experts.

    When ``num_layers`` is smaller than the backbone layer count, backbone KV
    layers are uniformly sampled so the expert sees a representative cross-section.
    """

    def __init__(
        self,
        model_name_or_path: str,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        use_adaln: bool = False,
        cond_hidden_size: int | None = None,
        detach_prefix_kv: bool = False,
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        use_kv_projection: bool = False,
    ):
        super().__init__()
        if use_adaln and cond_hidden_size is None:
            raise ValueError("cond_hidden_size is required when use_adaln=True.")
        if Qwen3VLConfig is None:
            raise ImportError(
                "Qwen3Expert requires transformers with Qwen3-VL support."
            )
        base = Qwen3VLConfig.from_pretrained(
            model_name_or_path,
        ).text_config

        config = Qwen3VLTextConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=base.num_key_value_heads,
            head_dim=base.head_dim,
            hidden_act=base.hidden_act,
            max_position_embeddings=base.max_position_embeddings,
            rms_norm_eps=base.rms_norm_eps,
            rope_parameters=base.rope_parameters,
            attention_bias=base.attention_bias,
            attention_dropout=getattr(base, "attention_dropout", 0.0),
            vocab_size=32,
        )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        elif hasattr(base, "_attn_implementation"):
            config._attn_implementation = base._attn_implementation

        self.config = config
        self.hidden_size = hidden_size
        self.use_adaln = use_adaln
        self.detach_prefix_kv = detach_prefix_kv
        self.num_layers = num_layers
        self.backbone_num_layers = base.num_hidden_layers
        self.kv_layer_indices = compute_kv_layer_indices(num_layers, base.num_hidden_layers)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        # All layers use layer_idx=0 because each layer gets its own
        # single-entry cache at runtime.  Using the same index everywhere
        # prevents torch.compile from recompiling for every distinct layer_idx.
        self.layers = nn.ModuleList(
            [
                DiTQwen3DecoderLayer(
                    config=config,
                    layer_idx=0,
                    attention_cls=Qwen3VLTextAttention,
                    mlp_cls=Qwen3VLTextMLP,
                    use_adaln=use_adaln,
                    cond_hidden_size=cond_hidden_size or 0,
                    use_kv_projection=use_kv_projection,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1
        self._gradient_checkpointing_func = partial(
            checkpoint, use_reentrant=False, preserve_rng_state=False,
        )

    def enable_gradient_checkpointing(self, every_n: int = 1) -> None:
        self.gradient_checkpointing = True
        self.checkpoint_every_n = every_n
        self._gradient_checkpointing_func = partial(
            checkpoint, use_reentrant=False, preserve_rng_state=False,
        )

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1

    @staticmethod
    def build_4d_attention_mask(
        full_attention_mask_bool: torch.Tensor,
        prefix_len: int,
        action_len: int,
        chunk_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build 4D additive attention mask equivalent to the flex_attention BlockMask.

        Returns [B, 1, action_len, prefix_len + action_len] with 0 for visible
        positions and a large negative value for masked positions.
        """
        device = full_attention_mask_bool.device
        kv_len = prefix_len + action_len

        # Per-KV validity: [B, 1, 1, KV]
        valid = full_attention_mask_bool[:, None, None, :]

        kv_indices = torch.arange(kv_len, device=device)
        q_indices = torch.arange(action_len, device=device)

        # Prefix positions are always visible: [1, 1, 1, KV]
        is_prefix = (kv_indices < prefix_len).view(1, 1, 1, -1)

        # Same-chunk logic for the action part: [1, 1, Q, KV]
        q_chunks = q_indices // chunk_size
        kv_action_offset = (kv_indices - prefix_len).clamp(min=0)
        kv_chunks = kv_action_offset // chunk_size
        same_chunk = q_chunks.view(1, 1, -1, 1) == kv_chunks.view(1, 1, 1, -1)
        is_action_kv = kv_indices >= prefix_len
        is_action_kv = is_action_kv.view(1, 1, 1, -1)

        # Reproduce mask_mod: valid & (is_prefix | (is_action_kv & same_chunk))
        bool_mask = valid & (is_prefix | (is_action_kv & same_chunk))

        min_val = torch.finfo(dtype).min
        return torch.where(bool_mask, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), min_val, dtype=dtype, device=device))

    @staticmethod
    def _prepare_suffix_position_ids(
        suffix_position_ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Split expert suffix ids into text ids and Qwen3-VL THW RoPE ids.

        Accepted shapes:
        - [B, L]: legacy text-style positions, expanded to [T,H,W]=same.
        - [3, B, L]: Qwen3-VL MRoPE ids in [temporal, height, width] order.
        - [4, B, L]: [text_pos, temporal, height, width].
        """
        suffix_position_ids = suffix_position_ids.to(device=device, dtype=torch.long)
        expected_2d = (batch_size, seq_len)

        if suffix_position_ids.ndim == 2:
            if tuple(suffix_position_ids.shape) != expected_2d:
                raise ValueError(
                    "Expected 2D suffix_position_ids with shape "
                    f"{expected_2d}, got {tuple(suffix_position_ids.shape)}."
                )
            text_position_ids = suffix_position_ids
            rope_position_ids = suffix_position_ids.unsqueeze(0).expand(3, -1, -1)
            return text_position_ids, rope_position_ids

        if suffix_position_ids.ndim != 3:
            raise ValueError(
                "Expected suffix_position_ids to be 2D [B,L], 3D [3,B,L], "
                f"or 3D [4,B,L], got {tuple(suffix_position_ids.shape)}."
            )
        if tuple(suffix_position_ids.shape[1:]) != expected_2d:
            raise ValueError(
                "Expected suffix_position_ids trailing shape "
                f"{expected_2d}, got {tuple(suffix_position_ids.shape[1:])}."
            )

        if suffix_position_ids.shape[0] == 3:
            return None, suffix_position_ids
        if suffix_position_ids.shape[0] == 4:
            return suffix_position_ids[0], suffix_position_ids[1:]
        raise ValueError(
            "Expected first suffix_position_ids dimension to be 3 "
            f"([temporal,height,width]) or 4 ([text,temporal,height,width]), "
            f"got {suffix_position_ids.shape[0]}."
        )

    def forward(
        self,
        suffix_embeds: torch.Tensor,
        prefix_cache: PrefixKVCache,
        suffix_position_ids: torch.Tensor,
        cond: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
        num_parallel_chunks: int = 1,
        output_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        if prefix_cache is None:
            raise ValueError("Expert requires a prefix cache.")
        if prefix_cache.num_layers < max(self.kv_layer_indices) + 1:
            raise ValueError(
                f"Prefix cache has {prefix_cache.num_layers} layers, "
                f"but expert needs layer index up to {max(self.kv_layer_indices)}."
            )
        if num_parallel_chunks < 1:
            raise ValueError(f"num_parallel_chunks must be >= 1, got {num_parallel_chunks}.")
        if suffix_mask is None:
            raise ValueError("suffix_mask is required. Pass a boolean mask for all expert tokens.")

        if self.detach_prefix_kv:
            prefix_cache = prefix_cache.detach()

        hidden_states = suffix_embeds
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Build full attention mask: [prefix_mask | suffix_mask]
        full_attention_mask = torch.cat(
            [
                prefix_cache.mask.to(device=suffix_mask.device, dtype=torch.bool),
                suffix_mask.to(dtype=torch.bool),
            ],
            dim=-1,
        )

        text_position_ids, rope_position_ids = self._prepare_suffix_position_ids(
            suffix_position_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
        )
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        prefix_len = prefix_cache.kv_seq_len
        suffix_len = seq_len
        if suffix_len % num_parallel_chunks != 0:
            raise ValueError(
                f"Suffix length {suffix_len} must be divisible by num_parallel_chunks={num_parallel_chunks}."
            )

        chunk_size = suffix_len // num_parallel_chunks
        full_attention_mask_bool = full_attention_mask.to(device=hidden_states.device)

        use_flex = self.config._attn_implementation == "flex_attention"
        if use_flex:
            from torch.nn.attention.flex_attention import create_block_mask

            def mask_mod(b, h, q_idx, kv_idx):
                del h
                valid = full_attention_mask_bool[b, kv_idx]
                is_prefix = kv_idx < prefix_len
                same_chunk = (q_idx // chunk_size) == ((kv_idx - prefix_len) // chunk_size)
                return valid & (is_prefix | same_chunk)

            # T=1 is the same code path with a single chunk, which degenerates to
            # fully bidirectional suffix attention plus always-visible prefix KV.
            attention_mask = create_block_mask(
                mask_mod=mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=suffix_len,
                KV_LEN=prefix_len + suffix_len,
                device=hidden_states.device,
            )
        else:
            # Eager / sdpa: build a standard 4D additive attention mask
            # that reproduces the same masking logic.
            attention_mask = self.build_4d_attention_mask(
                full_attention_mask_bool,
                prefix_len,
                suffix_len,
                chunk_size,
                dtype=hidden_states.dtype,
            )

        all_attn_weights: list[torch.Tensor | None] = []
        for layer_idx, layer in enumerate(self.layers):
            # Map expert layer to backbone KV layer (identity when counts match)
            backbone_layer = self.kv_layer_indices[layer_idx]
            prefix_key = prefix_cache.keys[backbone_layer]
            prefix_value = prefix_cache.values[backbone_layer]

            if (
                self.gradient_checkpointing
                and self.training
                and layer_idx % self.checkpoint_every_n == 0
            ):
                hidden_states, layer_attn = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    prefix_key,
                    prefix_value,
                    cond,
                    text_position_ids,
                    output_attentions,
                )
            else:
                hidden_states, layer_attn = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    prefix_key=prefix_key,
                    prefix_value=prefix_value,
                    cond=cond,
                    text_position_ids=text_position_ids,
                    output_attentions=output_attentions,
                )
            if output_attentions:
                all_attn_weights.append(layer_attn)

        hidden_states = self.norm(hidden_states)
        result = hidden_states * suffix_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_attentions:
            return result, all_attn_weights
        return result
