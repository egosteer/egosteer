from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
    create_causal_mask,
    eager_attention_forward,
)

from src.model.vlm.prefix_cache import BackboneStreamOutput
from src.model.vlm.qwen3_vl_compile_patch import apply_patch

# Patch Qwen3VLVisionAttention.forward at backbone import time so every
# entry point that touches the backbone (train / inference wrapper / serve)
# gets the fix for transformers issue #44962. Idempotent: safe to call
# multiple times; the function only rebinds Qwen3VLVisionAttention.forward.
apply_patch()


@dataclass
class BackboneEmbedOutput:
    inputs_embeds: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_visual_embeds: list[torch.Tensor] | None


class Qwen3VLTextAttentionWithKV(Qwen3VLTextAttention):
    """Wrap the upstream attention block and also expose the full-sequence key/value tensors."""

    def __init__(self, base_attention: Qwen3VLTextAttention):
        nn.Module.__init__(self)
        # Keep hidden ref for non-module attributes (config, head_dim, scaling, etc.)
        object.__setattr__(self, "base_attention", base_attention)
        # Register parameter-bearing submodules so FSDP2 can manage them.
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        base_attention = self.base_attention
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, base_attention.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                base_attention.layer_idx,
                cache_kwargs,
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base_attention.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            base_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            # Use self.training because base_attention is registered via
            # object.__setattr__ (line 37) and is not a real submodule, so
            # wrapper.eval() / .train() does not propagate into it.
            dropout=0.0 if not self.training else base_attention.attention_dropout,
            scaling=base_attention.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, key_states, value_states


class Qwen3VLTextDecoderLayerWithKV(Qwen3VLTextDecoderLayer):
    """Wrap the upstream decoder layer and return the layer key/value tensors explicitly."""

    def __init__(self, base_layer: Qwen3VLTextDecoderLayer):
        nn.Module.__init__(self)
        self.self_attn = Qwen3VLTextAttentionWithKV(base_layer.self_attn)
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.mlp = base_layer.mlp
        self.hidden_size = base_layer.hidden_size
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Second return value from attention_interface:
        #   eager  -> full [B,H,Q,K] attention matrix (always computed)
        #   flex   -> LSE tensor or None
        #   sdpa   -> None
        # We only collect it when the caller sets output_attentions=True
        # AND the backend is eager (enforced at the config level).
        hidden_states, attn_weights, key_states, value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights, key_states, value_states


class Qwen3VLTextModelWithKV(Qwen3VLTextModel):
    """Run the upstream decoder stack while returning full-sequence layer KV for prefix reuse.

    This wrapper keeps the original forward inputs intact and only changes the outputs:
    `past_key_values` is reused to carry one `(key, value)` tuple per decoder layer.
    """

    def __init__(self, base_model: Qwen3VLTextModel):
        nn.Module.__init__(self)
        object.__setattr__(self, "upstream_text_model", base_model)
        self.config = base_model.config
        self.padding_idx = base_model.padding_idx
        self.vocab_size = base_model.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayerWithKV(layer) for layer in base_model.layers]
        )
        # Clear original layers so FSDP2 doesn't find the same parameters
        # through both the original and wrapper module paths.
        base_model.layers = nn.ModuleList()
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1
        self._gradient_checkpointing_func = partial(
            checkpoint, use_reentrant=False, preserve_rng_state=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_dict = bool(kwargs.pop("return_dict", True))
        kwargs.pop("output_hidden_states", None)
        output_attentions = bool(kwargs.pop("output_attentions", False))

        base_model = self.upstream_text_model
        if inputs_embeds is None:
            inputs_embeds = base_model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = int(past_key_values.get_seq_length())
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Qwen3-VL position_ids: 4 dims = [text_pos, height, width, temporal]
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = base_model.rotary_emb(hidden_states, rope_position_ids)
        layer_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        all_attn_weights: list[torch.Tensor | None] = [] if output_attentions else []

        for layer_idx, decoder_layer in enumerate(self.layers):
            # output_attentions is NOT forwarded to the attention layer because
            # Qwen3-VL attention backends ignore it. The backend choice alone
            # decides whether the second return value is real weights (eager),
            # LSE (flex), or None (sdpa). We collect at this level instead.
            layer_kwargs = dict(
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if (
                self.gradient_checkpointing
                and self.training
                and layer_idx % self.checkpoint_every_n == 0
            ):
                hidden_states, attn_weights, key_states, value_states = (
                    self._gradient_checkpointing_func(
                        decoder_layer, hidden_states, **layer_kwargs,
                    )
                )
            else:
                hidden_states, attn_weights, key_states, value_states = decoder_layer(
                    hidden_states, **layer_kwargs,
                )
            layer_kv.append((key_states, value_states))
            if output_attentions:
                all_attn_weights.append(attn_weights)

            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = base_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = base_model.norm(hidden_states)
        full_layer_kv = tuple(layer_kv)
        if not return_dict:
            return hidden_states, full_layer_kv
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=full_layer_kv,
            attentions=tuple(all_attn_weights) if output_attentions else None,
        )


class Qwen3VLBackboneWrapper(nn.Module):
    """Wrap the HF Qwen3-VL backbone and expose the project-specific embedding pipeline."""

    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        freeze_backbone: bool = False,
        torch_dtype: str | None = None,
        state_token: str = "<state>",
        action_token: str = "<action>",
        camera_token: str = "",
        device_map: Any = None,
        low_cpu_mem_usage: bool = True,
        text_attn_implementation: str = "sdpa",
        vision_attn_implementation: str = "flash_attention_2",
        use_lora: bool = False,
        lora: Any = None,
        use_quantization: bool = False,
        quantization: Any = None,
    ):
        super().__init__()
        if use_lora:
            raise NotImplementedError("Qwen3VLBackboneWrapper does not support LoRA at inference time.")
        if use_quantization:
            raise NotImplementedError(
                "Qwen3VLBackboneWrapper does not support quantization at inference time."
            )
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL backbone requires a recent transformers installation."
            ) from exc

        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = processor.tokenizer
        special_tokens = [state_token, action_token]
        if camera_token:
            special_tokens.append(camera_token)
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        resolved_dtype = self._resolve_torch_dtype(torch_dtype)

        if not isinstance(text_attn_implementation, str):
            raise TypeError("text_attn_implementation must be a string.")
        if not isinstance(vision_attn_implementation, str):
            raise TypeError("vision_attn_implementation must be a string.")
        # Load with sdpa; runtime attention backend is set below via config._attn_implementation.
        # from_pretrained rejects "flex_attention" at the validation gate, but the actual
        # dispatch happens through ALL_ATTENTION_FUNCTIONS at forward time.
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation="sdpa",
        )
        self.model.resize_token_embeddings(len(tokenizer))

        text_config = self.model.config.text_config
        # base_model / hf_language_model / lm_head are exposed as properties (see
        # below) rather than submodule attributes. Assigning them as submodules
        # would register the same shared backbone weights under several keys,
        # duplicating them in state_dict / DCP checkpoints.
        self.language_model = Qwen3VLTextModelWithKV(self.hf_language_model)

        self.model.config.text_config._attn_implementation = text_attn_implementation
        self.model.config.vision_config._attn_implementation = vision_attn_implementation
        self.hf_language_model.config._attn_implementation = text_attn_implementation
        self.language_model.config._attn_implementation = text_attn_implementation
        self.base_model.model.visual.config._attn_implementation = vision_attn_implementation

        self.tokenizer = tokenizer
        self.processor = processor
        self.hidden_size = text_config.hidden_size
        self.vocab_size = len(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.image_token = getattr(processor, "image_token", "<|image_pad|>")
        self.video_token = getattr(processor, "video_token", "<|video_pad|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.state_token_id = tokenizer.convert_tokens_to_ids(state_token)
        self.action_token_id = tokenizer.convert_tokens_to_ids(action_token)
        self.camera_token_id = (
            tokenizer.convert_tokens_to_ids(camera_token) if camera_token else None
        )
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = text_config.num_key_value_heads
        self.head_dim = int(getattr(text_config, "head_dim", self.hidden_size // self.num_heads))

        if freeze_backbone:
            self.freeze_parameters()

    # Shortcut references exposed as properties (not submodules) so the shared
    # backbone weights are stored exactly once in state_dict, under `self.model`.
    @property
    def base_model(self):
        return self.model

    @property
    def hf_language_model(self):
        return self.base_model.model.language_model

    @property
    def lm_head(self):
        return self.model.lm_head

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str | torch.dtype | None) -> torch.dtype | None:
        if dtype_name is None:
            return None
        if isinstance(dtype_name, torch.dtype):
            return dtype_name
        return getattr(torch, str(dtype_name))

    def enable_gradient_checkpointing(
        self,
        text_every_n: int = 1,
        vision_every_n: int = 1,
    ) -> None:
        """Enable checkpointing on the vision tower and the training text wrapper.

        Args:
            text_every_n: checkpoint every N-th text layer. 0 disables.
            vision_every_n: checkpoint every N-th vision block. 0 disables.
                HF's GradientCheckpointingLayer checks a per-block boolean flag,
                so we first enable all blocks then selectively disable the
                non-selected ones.
        """
        checkpoint_func = partial(
            checkpoint, use_reentrant=False, preserve_rng_state=False,
        )
        if text_every_n > 0:
            self.language_model.gradient_checkpointing = True
            self.language_model.checkpoint_every_n = text_every_n
            self.language_model._gradient_checkpointing_func = checkpoint_func
            for layer in self.language_model.layers:
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = checkpoint_func

        if vision_every_n > 0:
            visual_model = self.base_model.model.visual
            enable_method = getattr(visual_model, "gradient_checkpointing_enable", None)
            if callable(enable_method):
                enable_method(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False,
                        "preserve_rng_state": False,
                    },
                )
            # Selective every-N: disable on non-selected blocks.
            if vision_every_n > 1 and hasattr(visual_model, "blocks"):
                for i, blk in enumerate(visual_model.blocks):
                    if i % vision_every_n != 0:
                        blk.gradient_checkpointing = False

    def disable_gradient_checkpointing(self) -> None:
        """Disable checkpointing on the vision tower and the training text wrapper."""
        self.language_model.gradient_checkpointing = False
        self.language_model.checkpoint_every_n = 1
        for layer in self.language_model.layers:
            layer.gradient_checkpointing = False

        visual_model = self.base_model.model.visual
        disable_method = getattr(visual_model, "gradient_checkpointing_disable", None)
        if callable(disable_method):
            disable_method()

    def _video_features_for_lm(
        self,
        video_outputs: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """ViT video output -> LM-shaped features: hand all T frames through.

        HF returns pooler_output as a per-entry tuple of (T*N, D) tensors and
        deepstack_features as flat (B*T*N, D) per layer.
        """
        return torch.cat(list(video_outputs.pooler_output), dim=0), video_outputs.deepstack_features

    def encode_visual_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        base_model = self.base_model
        has_images = pixel_values is not None
        has_videos = pixel_values_videos is not None

        # Mixed batch: one merged ViT call (typically only at inference).
        if has_images and has_videos:
            n_image_entries = image_grid_thw.shape[0]
            combined_pv = torch.cat([pixel_values, pixel_values_videos], dim=0)
            combined_grid = torch.cat([image_grid_thw, video_grid_thw], dim=0)

            combined_out = base_model.get_image_features(
                pixel_values=combined_pv,
                image_grid_thw=combined_grid,
                return_dict=True,
            )

            image_embeds = torch.cat(combined_out.pooler_output[:n_image_entries], dim=0)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            # Split deepstack along the image/video token boundary.
            sms = base_model.model.visual.spatial_merge_size
            img_tokens = (image_grid_thw.prod(dim=-1) // (sms * sms)).sum().item()
            deepstack_image = [f[:img_tokens] for f in combined_out.deepstack_features]
            deepstack_video_flat = [f[img_tokens:] for f in combined_out.deepstack_features]

            video_outputs_like = type("VideoOutputs", (), {})()
            video_outputs_like.pooler_output = combined_out.pooler_output[n_image_entries:]
            video_outputs_like.deepstack_features = deepstack_video_flat
            video_embeds, deepstack_video = self._video_features_for_lm(video_outputs_like)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask, _ = base_model.model.get_placeholder_mask(
                input_ids=input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            _, video_mask = base_model.model.get_placeholder_mask(
                input_ids=input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Merge deepstack features in text-sequence positional order.
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for img_feat, vid_feat in zip(deepstack_image, deepstack_video):
                merged = img_feat.new_zeros(visual_pos_masks.sum(), img_feat.shape[-1]).to(img_feat.device)
                merged[image_mask_joint, :] = img_feat
                merged[video_mask_joint, :] = vid_feat
                deepstack_visual_embeds.append(merged)

            return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

        # Image-only: pure VLM / inference path (no temporal context).
        if has_images:
            image_outputs = base_model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = base_model.model.get_placeholder_mask(
                input_ids=input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            return inputs_embeds, image_mask[..., 0], image_outputs.deepstack_features

        if has_videos:
            video_outputs = base_model.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_dict=True,
            )
            video_embeds, deepstack_features = self._video_features_for_lm(video_outputs)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = base_model.model.get_placeholder_mask(
                input_ids=input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            return inputs_embeds, video_mask[..., 0], deepstack_features

        return inputs_embeds, None, None

    def replace_slot_embeddings(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        state_slot_embeds: torch.Tensor | None,
        action_slot_embeds: torch.Tensor | None,
        camera_slot_embeds: torch.Tensor | None = None,
        state_token_id: int | None = None,
        action_token_id: int | None = None,
    ) -> torch.Tensor:
        state_token_id = self.state_token_id if state_token_id is None else state_token_id
        action_token_id = self.action_token_id if action_token_id is None else action_token_id

        if state_slot_embeds is not None:
            state_mask = (input_ids == state_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(state_mask, state_slot_embeds.to(inputs_embeds.dtype))

        if action_slot_embeds is not None:
            action_mask = (input_ids == action_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_slot_embeds.to(inputs_embeds.dtype))
        
        if camera_slot_embeds is not None and self.camera_token_id is not None:
            cam_mask = (input_ids == self.camera_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(cam_mask, camera_slot_embeds.to(inputs_embeds.dtype))

        return inputs_embeds

    def build_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        state_slot_embeds: torch.Tensor | None,
        action_slot_embeds: torch.Tensor | None,
        camera_slot_embeds: torch.Tensor | None = None,
        state_token_id: int | None = None,
        action_token_id: int | None = None,
    ) -> BackboneEmbedOutput:
        """Construct final language-model embeddings from project batch tensors."""
        del mm_token_type_ids
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self.encode_visual_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        inputs_embeds = self.replace_slot_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            state_slot_embeds=state_slot_embeds,
            action_slot_embeds=action_slot_embeds,
            camera_slot_embeds=camera_slot_embeds,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        return BackboneEmbedOutput(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        state_slot_embeds: torch.Tensor | None,
        action_slot_embeds: torch.Tensor | None,
        camera_slot_embeds: torch.Tensor | None = None,
        state_token_id: int | None = None,
        action_token_id: int | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        past_key_values: Any = None,
    ) -> BackboneStreamOutput:
        del output_hidden_states
        embed_output = self.build_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            state_slot_embeds=state_slot_embeds,
            action_slot_embeds=action_slot_embeds,
            camera_slot_embeds=camera_slot_embeds,
            state_token_id=state_token_id,
            action_token_id=action_token_id,
        )
        position_ids = self.base_model.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=embed_output.inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=embed_output.inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            visual_pos_masks=embed_output.visual_pos_masks,
            deepstack_visual_embeds=embed_output.deepstack_visual_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # In training, `past_key_values` carries one full-sequence `(key, value)` pair per text layer.
        return BackboneStreamOutput(
            last_hidden_states=outputs.last_hidden_state,
            position_ids=position_ids,
            past_key_values_hf=outputs.past_key_values,
            attention_weights=outputs.attentions,
        )
