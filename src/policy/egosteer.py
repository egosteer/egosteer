from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.utils.compile_utils import compile_module_list
from src.model.vlm.prefix_cache import (
    BackboneStreamOutput,
    slice_prefix_cache_from_full_kv,
)
from src.utils.sample_utils import generate_bernoulli_mask, generate_multi_span_mask


@dataclass(frozen=True)
class FlowConfig:
    sig_min: float = 0.001
    num_parallel_t: int = 1
    sampling: str = "beta"
    alpha: float = 1.5
    beta: float = 1.0
    num_inference_steps: int = 10


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    delay_strategy: str = "uniform"
    max_delay: int = 6


@dataclass(frozen=True)
class LossConfig:
    ce_loss_weight: float = 0.1
    flow_loss_weight: float = 1.0
    wm_loss_weight: float = 0.0


@dataclass(frozen=True)
class SpanMaskConfig:
    mask_ratio: float = 0.4
    mean_span_len: float = 4.0
    start_bias_alpha: float = 1.5
    p_no_mask: float = 0.1
    keep_last: bool = False
    prefer_early: bool = True


@dataclass(frozen=True)
class StateMaskConfig:
    # Per-frame independent Bernoulli masking for the short (6-frame) state
    # history. Span masking is meaningless at this length, so each frame is
    # masked independently instead.
    mask_prob: float = 0.75    # per-frame independent mask probability
    keep_last: bool = True     # never mask the current (last) frame
    p_no_mask: float = 0.05    # fraction of samples left fully unmasked


class WorldModelHead(nn.Module):
    """Query tokens + output projection. view_embed is added per view
    (head/chest); zero-init keeps single-view output bit-identical.
    """

    def __init__(self, n_queries: int, hidden_size: int, upsample_factor: int,
                 num_views: int = 2):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(n_queries, hidden_size) * 0.02)
        self.view_embed = nn.Parameter(torch.zeros(num_views, hidden_size))
        self.output_proj = nn.Linear(hidden_size, hidden_size * upsample_factor ** 2)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)


@dataclass(frozen=True)
class WorldModelConfig:
    num_future_frames: int = 0
    target_image_size: tuple[int, int] | None = None
    teacher_patch_size: int = 16
    upsample_factor: int = 2
    action_conditioning: bool = False
    motion_conditioning: bool = False
    mask_loss_by_view_mask: bool = True


@dataclass(frozen=True)
class ARActionTrainConfig:
    input_mask_enabled: bool = False
    state_mask: StateMaskConfig = StateMaskConfig()


class InputMaskEmbeddings(nn.Module):
    """Learnable vector for span-masking state slot embeddings."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.state = nn.Parameter(torch.randn(hidden_size) * 0.02)


class EgoSteer(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        time_embedding: nn.Module,
        flow_expert: nn.Module,
        action_decoder: nn.Module,
        shape_meta: dict,
        ignore_index: int = -100,
        action_hidden_size: int = 1024,
        flow_config: FlowConfig = FlowConfig(),
        rtc_config: RTCConfig = RTCConfig(),
        loss_config: LossConfig = LossConfig(),
        ar_action_train_config: ARActionTrainConfig = ARActionTrainConfig(),
        # Camera intrinsic as token embedding (optional)
        camera_intrinsic_mode: str = "text",
        camera_encoder: nn.Module | None = None,
        # World model (optional)
        world_model_expert: nn.Module | None = None,
        frozen_teacher: nn.Module | None = None,
        world_model_config: WorldModelConfig = WorldModelConfig(),
        wm_action_encoder: nn.Module | None = None,
        wm_motion_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.shape_meta = shape_meta

        # Camera intrinsic encoding
        self.camera_intrinsic_mode = camera_intrinsic_mode
        self.camera_encoder = camera_encoder

        # Backbone and derived attributes
        self.backbone = backbone
        self.vlm_hidden_size = int(getattr(self.backbone, "hidden_size"))
        self.vocab_size = int(getattr(self.backbone, "vocab_size"))
        self.pad_token_id = int(getattr(self.backbone, "pad_token_id"))
        self.image_token_index = int(getattr(self.backbone, "image_token_id"))
        self.state_token_index = int(getattr(self.backbone, "state_token_id"))
        self.action_token_index = int(getattr(self.backbone, "action_token_id"))
        self.eos_token_id = getattr(getattr(self.backbone, "tokenizer", None), "eos_token_id", None)

        # Grouped config (frozen dataclasses)
        self.ignore_index = ignore_index
        self.CELoss = nn.CrossEntropyLoss(reduction="sum", ignore_index=ignore_index)
        self.flow_config = flow_config
        self.rtc_config = rtc_config
        self.loss_config = loss_config
        self.ar_action_train_config = ar_action_train_config

        if self.flow_config.num_parallel_t < 1:
            raise ValueError(f"num_parallel_t must be >= 1, got {self.flow_config.num_parallel_t}.")
        if self.flow_config.sampling not in ("beta", "uniform"):
            raise ValueError(f"Unsupported flow sampling strategy: {self.flow_config.sampling}")
        # Mutable: overridden at inference time by egosteer_inference_wrapper.
        self.num_inference_steps = self.flow_config.num_inference_steps

        # Shape meta derived
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.state_dim = int(shape_meta["obs"]["state"]["shape"][0])
        self.horizon_steps = int(shape_meta["action"]["horizon"])
        self.num_state_tokens = int(shape_meta["obs"]["state"]["horizon"])
        self.num_action_tokens = int(shape_meta["action"]["horizon"])
        self.action_horizon = self.num_action_tokens

        # Expert hidden size (explicit parameter, must match action_encoder output width)
        self.action_hidden_size = action_hidden_size

        # Submodules (all pre-instantiated by Hydra)
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.time_embedding = time_embedding
        self.flow_expert = flow_expert
        self.action_decoder = action_decoder

        # State-slot mask embedding, only when input masking is enabled.
        if self.ar_action_train_config.input_mask_enabled:
            self.input_mask_embeddings = InputMaskEmbeddings(self.vlm_hidden_size)

        # World model components
        self.world_model_expert = world_model_expert
        self.frozen_teacher = frozen_teacher
        self.world_model_config = world_model_config
        self.wm_action_encoder = wm_action_encoder
        self.wm_motion_encoder = wm_motion_encoder
        self.use_world_model = world_model_expert is not None and frozen_teacher is not None
        if self.use_world_model:
            self._init_world_model(world_model_expert, world_model_config)
            if world_model_config.action_conditioning and wm_action_encoder is None:
                raise ValueError(
                    "world_model.action_conditioning=True requires wm_action_encoder"
                )
            if world_model_config.motion_conditioning and wm_motion_encoder is None:
                raise ValueError(
                    "world_model.motion_conditioning=True requires wm_motion_encoder"
                )
            if wm_action_encoder is not None and not world_model_config.action_conditioning:
                for param in wm_action_encoder.parameters():
                    param.requires_grad = False
            if wm_motion_encoder is not None and not world_model_config.motion_conditioning:
                for param in wm_motion_encoder.parameters():
                    param.requires_grad = False

    def _init_world_model(self, expert: nn.Module, config: WorldModelConfig) -> None:
        tH, tW = config.target_image_size
        stride = config.teacher_patch_size * config.upsample_factor
        assert tH % stride == 0 and tW % stride == 0, (
            f"target ({tH},{tW}) must be divisible by patch*upsample={stride}"
        )
        self.wm_grid_h = tH // stride
        self.wm_grid_w = tW // stride
        self.wm_upsample_factor = config.upsample_factor
        self.wm_num_future_frames = config.num_future_frames

        D = expert.hidden_size
        n_queries = config.num_future_frames * self.wm_grid_h * self.wm_grid_w
        self.wm_head = WorldModelHead(n_queries, D, config.upsample_factor)

    def compile_blocks(
        self,
        compile_kwargs: dict[str, Any],
    ) -> None:
        compile_flags = self.resolve_compile_block_flags(compile_kwargs)
        block_compile_kwargs = {
            key: value
            for key, value in compile_kwargs.items()
            if key not in {"vision", "text", "flow", "world_model"}
        }

        if compile_flags["vision"]:
            compile_module_list(self.backbone.base_model.model.visual.blocks, block_compile_kwargs)
        if compile_flags["text"]:
            compile_module_list(self.backbone.language_model.layers, block_compile_kwargs)
        if compile_flags["flow"]:
            compile_module_list(self.flow_expert.layers, block_compile_kwargs)
        if compile_flags["world_model"] and self.use_world_model:
            compile_module_list(self.world_model_expert.layers, block_compile_kwargs)
            # DINOv3 ViT layer list lives at `model.model.layer`.
            compile_module_list(self.frozen_teacher.model.model.layer, block_compile_kwargs)

    def resolve_compile_block_flags(
        self,
        compile_kwargs: dict[str, Any],
    ) -> dict[str, bool]:
        return {
            "vision": bool(compile_kwargs.get("vision", True)),
            "text": bool(compile_kwargs.get("text", True)),
            "flow": bool(compile_kwargs.get("flow", True)),
            "world_model": bool(compile_kwargs.get("world_model", True)),
        }

    def enable_gradient_checkpointing(self, config: bool | dict | None = None) -> None:
        """Enable checkpointing on modules that support it.

        Args:
            config: per-component checkpointing config. Accepted forms:
                - True / None: fully checkpoint every component (every_n=1).
                - False: no-op.
                - dict with optional keys ``text``, ``vision``, ``action_expert``;
                  each value is ``{enabled: bool, every_n: int}``. A dict
                  whose components are all disabled / missing is a no-op.
        """
        if config is False:
            return
        if config is None or config is True:
            config = {}

        text_cfg = config.get("text", {})
        vision_cfg = config.get("vision", {})
        expert_cfg = config.get("action_expert", {})

        # Defaults: enabled=True, every_n=1 (same as the old behaviour).
        # every_n=0 disables checkpointing for that component.
        text_enabled = text_cfg.get("enabled", True)
        text_every_n = text_cfg.get("every_n", 1) if text_enabled else 0
        vision_enabled = vision_cfg.get("enabled", True)
        vision_every_n = vision_cfg.get("every_n", 1) if vision_enabled else 0
        expert_enabled = expert_cfg.get("enabled", True)
        expert_every_n = expert_cfg.get("every_n", 1) if expert_enabled else 0

        if text_every_n == 0 and vision_every_n == 0 and expert_every_n == 0:
            return

        self.backbone.enable_gradient_checkpointing(
            text_every_n=text_every_n,
            vision_every_n=vision_every_n,
        )
        if expert_every_n > 0:
            self.flow_expert.enable_gradient_checkpointing(every_n=expert_every_n)

    def disable_gradient_checkpointing(self) -> None:
        """Disable checkpointing on modules that support it."""
        for module in (self.backbone, self.flow_expert):
            disable_method = getattr(module, "disable_gradient_checkpointing", None)
            if callable(disable_method):
                disable_method()

    @property
    def trainable_vlm_parameters(self):
        return [param for param in self.backbone.parameters() if param.requires_grad]

    @property
    def trainable_vision_parameters(self):
        visual = getattr(self.backbone.base_model.model, "visual", None)
        if visual is None:
            return []
        return [param for param in visual.parameters() if param.requires_grad]

    @property
    def lm_head(self):
        # Property (not a submodule) so the lm_head weight is not stored a second
        # time under the policy's top level; it lives under the backbone.
        return self.backbone.lm_head

    @property
    def trainable_text_parameters(self):
        # Language model + lm_head + embedding; exclude vision tower
        vision_ids = {id(p) for p in (getattr(self.backbone.base_model.model, "visual", None) or nn.Module()).parameters()}
        return [param for param in self.backbone.parameters() if param.requires_grad and id(param) not in vision_ids]

    @property
    def action_expert_parameters(self):
        modules = [
            self.action_encoder,
            self.time_embedding,
            self.flow_expert,
            self.action_decoder,
        ]
        camera_encoder = getattr(self, "camera_encoder", None)
        if camera_encoder is not None:
            modules.append(camera_encoder)
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    @property
    def ar_action_heads_parameters(self):
        # Heads attached to the VLM backbone. state_encoder is always here
        # (backbone always consumes state).
        modules = [self.state_encoder]
        if hasattr(self, "input_mask_embeddings"):
            modules.append(self.input_mask_embeddings)
        return [param for module in modules for param in module.parameters() if param.requires_grad]

    @property
    def world_model_parameters(self):
        params = []
        if self.use_world_model:
            params.extend(p for p in self.world_model_expert.parameters() if p.requires_grad)
            params.extend(p for p in self.wm_head.parameters() if p.requires_grad)
            # Gate by the conditioning flags: if the forward path won't call
            # the encoder, don't put its params in the optimizer — otherwise
            # fused AdamW never sees a gradient, never creates `step` state,
            # and checkpoint load fails on the missing key.
            if self.wm_action_encoder is not None and self.world_model_config.action_conditioning:
                params.extend(p for p in self.wm_action_encoder.parameters() if p.requires_grad)
            if self.wm_motion_encoder is not None and self.world_model_config.motion_conditioning:
                params.extend(p for p in self.wm_motion_encoder.parameters() if p.requires_grad)
        return params

    def build_prefix_lengths(self, batch: dict) -> torch.Tensor:
        answer_start_idx = batch.get("answer_start_idx")
        assert answer_start_idx is not None, (
            "answer_start_idx is required in batch. "
            "Ensure the collator provides it."
        )
        return answer_start_idx.to(device=batch["input_ids"].device, dtype=torch.long)

    def build_suffix_mrope_base(
        self,
        batch: dict,
        backbone_output: BackboneStreamOutput,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return the Qwen3-VL MRoPE coordinate where expert suffix tokens start.

        Qwen3-VL advances multimodal positions by the maximum THW coordinate,
        not by raw token count. The expert suffix must therefore continue from
        the maximum valid prefix MRoPE id instead of ``answer_start_idx``.
        """
        if device is None:
            device = batch["input_ids"].device

        fallback = self.build_prefix_lengths(batch).to(device=device, dtype=torch.long)
        position_ids = backbone_output.position_ids
        if position_ids is None:
            return fallback

        position_ids = position_ids.to(device=device, dtype=torch.long)
        if position_ids.ndim == 2:
            rope_position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            rope_position_ids = position_ids
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            rope_position_ids = position_ids[1:]
        else:
            raise ValueError(
                "Expected backbone position_ids to be [B,S], [3,B,S], or [4,B,S], "
                f"got {tuple(position_ids.shape)}."
            )

        prefix_lengths = self.build_prefix_lengths(batch).to(device=device, dtype=torch.long)
        batch_size, seq_len = rope_position_ids.shape[1], rope_position_ids.shape[2]
        if prefix_lengths.shape[0] != batch_size:
            raise ValueError(
                "Prefix length batch size does not match backbone position ids: "
                f"{prefix_lengths.shape[0]} vs {batch_size}."
            )
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        prefix_mask = positions < prefix_lengths.unsqueeze(1)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            if attention_mask.shape != prefix_mask.shape:
                raise ValueError(
                    "attention_mask shape does not match backbone position ids: "
                    f"{tuple(attention_mask.shape)} vs {tuple(prefix_mask.shape)}."
                )
            prefix_mask = prefix_mask & attention_mask

        has_prefix = prefix_mask.any(dim=1)
        masked_positions = rope_position_ids.masked_fill(~prefix_mask.unsqueeze(0), 0)
        mrope_base = masked_positions.amax(dim=(0, 2)) + 1
        return torch.where(has_prefix, mrope_base, fallback)

    @staticmethod
    def _build_text_mrope_position_ids(
        text_start: int,
        rope_start: int,
        length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, int]:
        """Build [text, temporal, height, width] ids for text-style tokens."""
        text_pos = torch.arange(text_start, text_start + length, device=device, dtype=torch.long)
        rope_pos = torch.arange(rope_start, rope_start + length, device=device, dtype=torch.long)
        position_ids = torch.stack([text_pos, rope_pos, rope_pos, rope_pos], dim=0)
        return position_ids, text_start + length, rope_start + length

    @staticmethod
    def _build_vision_mrope_position_ids(
        text_start: int,
        rope_start: int,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, int]:
        """Build Qwen3-VL-style [text, temporal, height, width] ids for frame grids.

        Each future frame is treated like one Qwen3-VL visual block with
        THW order [temporal, height, width]. The MRoPE cursor advances by
        ``max(grid_h, grid_w)`` per frame, matching Qwen3-VL's visual span
        rule, while text ids remain monotonic over the flattened suffix tokens.
        """
        if num_frames < 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError(
                f"Invalid vision MRoPE shape: num_frames={num_frames}, grid_h={grid_h}, grid_w={grid_w}."
            )
        frame_len = grid_h * grid_w
        h_offsets = torch.arange(grid_h, device=device, dtype=torch.long).repeat_interleave(grid_w)
        w_offsets = torch.arange(grid_w, device=device, dtype=torch.long).repeat(grid_h)

        segments: list[torch.Tensor] = []
        text_pos = text_start
        rope_pos = rope_start
        frame_span = max(grid_h, grid_w)
        for _ in range(num_frames):
            text_ids = torch.arange(text_pos, text_pos + frame_len, device=device, dtype=torch.long)
            temporal_ids = torch.full((frame_len,), rope_pos, device=device, dtype=torch.long)
            height_ids = rope_pos + h_offsets
            width_ids = rope_pos + w_offsets
            segments.append(torch.stack([text_ids, temporal_ids, height_ids, width_ids], dim=0))
            text_pos += frame_len
            rope_pos += frame_span

        if segments:
            position_ids = torch.cat(segments, dim=1)
        else:
            position_ids = torch.empty(4, 0, device=device, dtype=torch.long)
        return position_ids, text_pos, rope_pos

    def build_suffix_position_ids_from_relative(
        self,
        batch: dict,
        backbone_output: BackboneStreamOutput,
        rel_position_ids: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convert relative [text,T,H,W] ids to absolute Qwen3-VL suffix ids.

        The text dimension follows the token sequence and therefore starts at
        ``answer_start_idx``. The THW dimensions follow Qwen3-VL MRoPE
        coordinates and therefore continue from the maximum valid prefix MRoPE
        coordinate.
        """
        if device is None:
            device = rel_position_ids.device
        if rel_position_ids.ndim != 2 or rel_position_ids.shape[0] != 4:
            raise ValueError(
                "Expected rel_position_ids with shape [4, suffix_len], "
                f"got {tuple(rel_position_ids.shape)}."
            )

        rel_position_ids = rel_position_ids.to(device=device, dtype=torch.long)
        batch_size = batch["input_ids"].shape[0]
        text_base = self.build_prefix_lengths(batch).to(device=device, dtype=torch.long).view(1, batch_size, 1)
        rope_base = self.build_suffix_mrope_base(batch, backbone_output, device=device).view(1, batch_size, 1)

        position_ids = rel_position_ids.unsqueeze(1).expand(-1, batch_size, -1).clone()
        position_ids[:1] = position_ids[:1] + text_base
        position_ids[1:] = position_ids[1:] + rope_base
        return position_ids

    def build_action_position_ids(
        self,
        batch: dict,
        backbone_output: BackboneStreamOutput,
        action_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Build prefix-continuous [text, temporal, height, width] ids for action tokens.

        Args:
            batch: Collated batch (used for prefix masks).
            backbone_output: Backbone output carrying Qwen3-VL prefix position ids.
            action_ref: Any tensor with shape [B, action_len, ...] to derive dimensions from
                (e.g. action_embeds in flow stream, or raw actions in training).
        """
        action_len = action_ref.shape[1]
        device = action_ref.device
        rel_pos, _, _ = self._build_text_mrope_position_ids(
            text_start=0,
            rope_start=0,
            length=action_len,
            device=device,
        )
        return self.build_suffix_position_ids_from_relative(
            batch,
            backbone_output,
            rel_pos,
            device=device,
        )

    def build_slot_embeddings(self, batch: dict) -> dict[str, torch.Tensor | None]:
        slot_embeds: dict[str, torch.Tensor | None] = {
            "state": None, "action": None, "camera": None,
        }

        if self.camera_intrinsic_mode == "token" and self.camera_encoder is not None and "camera_intrinsic" in batch:
            camera_embeds = self.camera_encoder(batch["camera_intrinsic"])
            slot_embeds["camera"] = camera_embeds

        mask_cfg = self.ar_action_train_config
        do_mask = self.training and mask_cfg.input_mask_enabled

        if "states" in batch:
            state_embeds = self.state_encoder(batch["states"])  # [B, H_s, vlm_hidden_size]
            if do_mask:
                state_mask = generate_bernoulli_mask(
                    state_embeds.shape[0], self.num_state_tokens,
                    mask_cfg.state_mask, device=state_embeds.device,
                )
                state_embeds = torch.where(
                    state_mask.unsqueeze(-1),
                    self.input_mask_embeddings.state,
                    state_embeds,
                )
            slot_embeds["state"] = state_embeds

        return slot_embeds

    def forward_backbone_stream(
        self, batch: dict, slot_embeds: dict, output_attentions: bool = False,
    ) -> BackboneStreamOutput:
        output = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            state_slot_embeds=slot_embeds.get("state"),
            action_slot_embeds=slot_embeds.get("action"),
            camera_slot_embeds=slot_embeds.get("camera"),
            output_attentions=output_attentions,
        )
        output.prefix_cache = slice_prefix_cache_from_full_kv(
            output.past_key_values_hf,
            self.build_prefix_lengths(batch),
        )
        # KV cache is never detached at the backbone level.
        # Each expert controls its own detach via detach_prefix_kv.
        output.past_key_values_hf = None
        return output

    def forward_flow_stream(
        self,
        batch: dict,
        backbone_output: BackboneStreamOutput,
        flow_inputs: dict,
        num_parallel_chunks: int,
        output_attentions: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        time_for_model = flow_inputs["time_for_model"]
        if time_for_model.ndim != 2:
            raise ValueError(
                f"Expected packed flow time tensor with shape [B, T*H], got {time_for_model.shape}."
            )
        if num_parallel_chunks < 1:
            raise ValueError(f"num_parallel_chunks must be >= 1, got {num_parallel_chunks}.")

        batch_size, seq_len = time_for_model.shape
        time_cond = self.time_embedding(time_for_model.reshape(-1)).reshape(
            batch_size, seq_len, -1,
        )
        action_embeds = self.action_encoder(flow_inputs["noisy_actions"])
        action_mask = batch["actions_valid_mask"].any(dim=-1).to(dtype=torch.bool)
        # actions_valid_mask is always single-chunk [B, H, D]; position ids are
        # built per-chunk then repeated, matching the expanded action_embeds.
        action_position_ids = self.build_action_position_ids(
            batch,
            backbone_output,
            batch["actions_valid_mask"],
        )
        action_mask = action_mask.repeat(1, num_parallel_chunks)
        action_position_ids = action_position_ids.repeat(1, 1, num_parallel_chunks)
        if action_embeds.shape[1] != action_mask.shape[1]:
            raise ValueError(
                f"Action mask length {action_mask.shape[1]} does not match action embeddings length {action_embeds.shape[1]}."
            )
        if action_embeds.shape[1] != action_position_ids.shape[-1]:
            raise ValueError(
                "Action position ids length "
                f"{action_position_ids.shape[-1]} does not match action embeddings length {action_embeds.shape[1]}."
            )
        prefix_cache = backbone_output.prefix_cache
        expert_output = self.flow_expert(
            suffix_embeds=action_embeds,
            prefix_cache=prefix_cache,
            suffix_position_ids=action_position_ids,
            cond=time_cond,
            suffix_mask=action_mask,
            num_parallel_chunks=num_parallel_chunks,
            output_attentions=output_attentions,
        )
        if output_attentions:
            expert_hidden, expert_attn_weights = expert_output
        else:
            expert_hidden = expert_output
            expert_attn_weights = None
        pred_v = self.action_decoder(expert_hidden)
        return {
            "time_cond": time_cond,
            "action_hidden_states": expert_hidden,
            "pred_v": pred_v,
            "expert_attention_weights": expert_attn_weights,
        }

    def forward_world_model_stream(
        self,
        batch: dict,
        backbone_output: BackboneStreamOutput,
    ) -> dict[str, torch.Tensor]:
        """Run world model expert and frozen teacher on future frames.

        ``view_mask`` is the sample-level switch and the canonical view order
        is fixed to head, chest. Inactive view motion and query segments are
        present but masked out, and only active views contribute to loss.
        """
        cfg = self.world_model_config
        B = backbone_output.last_hidden_states.shape[0]
        device = backbone_output.last_hidden_states.device
        K = self.wm_num_future_frames
        gh, gw, uf = self.wm_grid_h, self.wm_grid_w, self.wm_upsample_factor
        D = self.world_model_expert.hidden_size
        view_mask = batch["view_mask"].to(device=device, dtype=torch.bool)
        if view_mask.ndim != 2 or view_mask.shape != (B, 2):
            raise ValueError(f"view_mask must have shape [B, 2], got {tuple(view_mask.shape)}")
        V = 2

        n_future = batch["n_future_frames"].to(device=device)
        frame_valid = torch.arange(K, device=device).unsqueeze(0) < n_future.unsqueeze(1)
        # Each of the K future steps owns gh*gw query tokens; replicate the
        # per-step validity across that block then flatten to [B, K*gh*gw].
        query_mask = frame_valid.unsqueeze(-1).expand(-1, -1, gh * gw).reshape(B, -1)

        # Build suffix with per-segment Qwen3-VL position ids. Text-like
        # conditioning tokens use T=H=W; future-frame query blocks use visual
        # [temporal, height, width] grids and advance in sequence for each view.
        segments: list[torch.Tensor] = []
        seg_masks: list[torch.Tensor] = []
        seg_position_ids: list[torch.Tensor] = []
        text_cur_pos = 0
        rope_cur_pos = 0

        if cfg.action_conditioning and "actions" in batch:
            action_cond = self.wm_action_encoder(batch["actions"])
            action_len = action_cond.shape[1]
            segments.append(action_cond)
            seg_masks.append(batch["actions_valid_mask"].any(dim=-1).to(dtype=torch.bool))
            pos_ids, text_cur_pos, rope_cur_pos = self._build_text_mrope_position_ids(
                text_start=text_cur_pos,
                rope_start=rope_cur_pos,
                length=action_len,
                device=device,
            )
            seg_position_ids.append(pos_ids)

        if cfg.motion_conditioning:
            segments.append(self.wm_motion_encoder(batch["future_head_motion"]))
            seg_masks.append(frame_valid & view_mask[:, 0:1])
            pos_ids, text_cur_pos, rope_cur_pos = self._build_text_mrope_position_ids(
                text_start=text_cur_pos,
                rope_start=rope_cur_pos,
                length=K,
                device=device,
            )
            seg_position_ids.append(pos_ids)
            segments.append(self.wm_motion_encoder(batch["future_chest_motion"]))
            seg_masks.append(frame_valid & view_mask[:, 1:2])
            pos_ids, text_cur_pos, rope_cur_pos = self._build_text_mrope_position_ids(
                text_start=text_cur_pos,
                rope_start=rope_cur_pos,
                length=K,
                device=device,
            )
            seg_position_ids.append(pos_ids)

        base_queries = self.wm_head.query_embed.unsqueeze(0).expand(B, -1, -1)
        query_len = base_queries.shape[1]
        for v in range(V):
            segments.append(base_queries + self.wm_head.view_embed[v])
            seg_masks.append(query_mask & view_mask[:, v:v + 1])
            pos_ids, text_cur_pos, rope_cur_pos = self._build_vision_mrope_position_ids(
                text_start=text_cur_pos,
                rope_start=rope_cur_pos,
                num_frames=K,
                grid_h=gh,
                grid_w=gw,
                device=device,
            )
            if pos_ids.shape[1] != query_len:
                raise ValueError(
                    f"World-model query position length {pos_ids.shape[1]} "
                    f"does not match query length {query_len}."
                )
            seg_position_ids.append(pos_ids)

        suffix = torch.cat(segments, dim=1)
        suffix_mask = torch.cat(seg_masks, dim=1)
        rel_position_ids = torch.cat(seg_position_ids, dim=1)  # [4, suffix_len]
        position_ids = self.build_suffix_position_ids_from_relative(
            batch,
            backbone_output,
            rel_position_ids,
            device=device,
        )
        if position_ids.shape[-1] != suffix.shape[1]:
            raise ValueError(
                f"World-model position ids length {position_ids.shape[-1]} "
                f"does not match suffix length {suffix.shape[1]}."
            )

        hidden = self.world_model_expert(
            suffix_embeds=suffix,
            prefix_cache=backbone_output.prefix_cache,
            suffix_position_ids=position_ids,
            suffix_mask=suffix_mask,
        )

        # Query block sits at the suffix tail; V views packed contiguously.
        query_hidden = hidden[:, -V * query_len:, :].reshape(B, V, query_len, D)
        x = self.wm_head.output_proj(query_hidden).reshape(B * V * K, gh, gw, uf, uf, D)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * V * K, gh * uf, gw * uf, D)
        pred = x.flatten(1, 2).reshape(B, V, K, -1, D)

        # Batch the teacher along B so DINOv3 only runs once.
        stacked = torch.cat([batch["future_frames"], batch["chest_future_frames"]], dim=0)
        head_t, chest_t = self.frozen_teacher(stacked).chunk(2, dim=0)
        target = torch.stack([head_t, chest_t], dim=1)

        return {
            "pred": pred,
            "target": target,
            "n_future_frames": batch["n_future_frames"],
            "view_mask": view_mask,
        }

    def compute_loss(self, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        from src.policy.egosteer_loss import compute_total_loss

        return compute_total_loss(self, batch)

    def freeze_vlm_weights(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_all_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_final_lm_norm(self) -> None:
        # final RMSNorm only participates in last_hidden_states (CE loss).
        # Without VLM data and without heads consuming
        # last_hidden_states, it never sees a grad, Adam skips state allocation,
        # and DCP metadata drops `exp_avg`/`exp_avg_sq`/`step` for it — which
        # then blocks resume.
        self.backbone.hf_language_model.norm.weight.requires_grad_(False)

    def infer_action(self, input: dict, **kwargs):
        from src.policy.egosteer_inference import infer_flow_action

        return infer_flow_action(self, input, **kwargs)

    def forward(self, mode: str, batch: dict, **kwargs) -> dict[str, torch.Tensor]:
        """Dispatch the top-level EgoSteer execution modes.

        Mode contract:
        - `train`: full multitask loss on one collated batch.
        - `infer_action`: continuous action inference from a prepared VLA batch.

        All modes expect the batch schema produced by the collator and backbone wrappers in this repository rather
        than raw Hugging Face model inputs alone.
        """
        if mode == "train":
            return self.compute_loss(batch, **kwargs)
        if mode == "infer_action":
            return self.infer_action(batch, **kwargs)
        raise ValueError(f"Invalid mode: {mode}")
