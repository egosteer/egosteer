from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger(__name__)


def ensure_uint8_vision_tensor(images: torch.Tensor, field_name: str) -> None:
    """Reject normalized float inputs because this batching path expects raw uint8 pixels."""
    if images.dtype != torch.uint8:
        raise ValueError(f"{field_name} must use dtype torch.uint8 before Qwen3-VL batching, got {images.dtype}")


def count_images(images: torch.Tensor) -> int:
    """Count image placeholders for one sample.

    Contract:
    - `images` is `[H, W, C]` for one image or `[T, H, W, C]` for multiple images.
    """
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[0])
    raise ValueError(f"Unsupported image container shape: {tuple(images.shape)}")


def build_sample_images(images: torch.Tensor) -> list[torch.Tensor]:
    """Convert one image sample into a normalized image batch for the HF processor.

    The returned value is always a list of image frames so mixed image batches do not need
    placeholder empty lists or mixed tensor/list top-level containers.
    Raw visual inputs must stay in `torch.uint8`; normalized float images should fail here.
    """
    ensure_uint8_vision_tensor(images, field_name="images")
    if images.ndim == 3:
        return [images]
    if images.ndim == 4:
        return [frame for frame in images]
    raise ValueError(f"Unsupported image container shape: {tuple(images.shape)}")


def build_sample_video(video: torch.Tensor) -> torch.Tensor:
    """Convert one video sample into the structure expected by the HF processor.

    Raw video frames must stay in `torch.uint8`; normalized float videos should fail here.
    """
    ensure_uint8_vision_tensor(video, field_name="video")
    if video.ndim != 4:
        raise ValueError(f"Unsupported video container shape: {tuple(video.shape)}")
    return video


def build_video_metadata(video: torch.Tensor, video_fps: float) -> dict[str, Any]:
    """Build video metadata for already-sampled frames.

    Contract:
    - `video` is `[T, H, W, C]`.
    - `video_fps` is the effective FPS after dataset stride sampling.
    """
    frame_count = int(video.shape[0])
    return {
        "total_num_frames": frame_count,
        "fps": float(video_fps),
        "frames_indices": list(range(frame_count)),
    }


def resolve_active_views(sample: dict[str, Any]) -> list[str]:
    """Return active video views in prompt/input order."""
    if "active_views" not in sample:
        raise KeyError("Video samples must provide active_views.")
    views = [str(view) for view in sample["active_views"]]
    valid = {"head", "chest"}
    invalid = [view for view in views if view not in valid]
    if invalid:
        raise ValueError(f"Unsupported active_views entries: {invalid}")
    if not views:
        raise ValueError("active_views must contain at least one view.")
    return views


class Qwen3VLChatFormatter:
    """Convert one project sample into the HF chat message structure expected by Qwen3-VL."""

    def __init__(
        self,
        state_token: str = "<state>",
        action_token: str = "<action>",
        camera_token: str = "<camera>",
        camera_intrinsic_mode: str = "text",
        predict_future_frames: bool = False,
    ):
        self.state_token = state_token
        self.action_token = action_token
        self.camera_token = camera_token
        self.camera_intrinsic_mode = camera_intrinsic_mode
        self.predict_future_frames = predict_future_frames

    def build_visual_content(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        vision_type = sample["vision_type"]
        if vision_type == "video":
            return [{"type": "video"} for _ in resolve_active_views(sample)]
        if vision_type == "image":
            return [{"type": "image"} for _ in range(count_images(sample["images"]))]
        raise ValueError(f"Unsupported vision_type: {vision_type}")

    def format_intrinsic_part(self, label: str, intrinsic: torch.Tensor) -> str:
        if self.camera_intrinsic_mode == "token":
            return f"{label} camera intrinsics: {self.camera_token}."
        values = intrinsic.tolist()
        return (
            f"{label} camera intrinsics: fx:{values[0]:.2f} fy:{values[1]:.2f} "
            f"cx:{values[2]:.2f} cy:{values[3]:.2f}."
        )

    def format_video_order_part(self, active_views: list[str]) -> str:
        labels = {"head": "head camera", "chest": "chest camera"}
        if len(active_views) == 1:
            return f"The video is from the {labels[active_views[0]]}."
        ordinals = ["first", "second"]
        parts = [
            f"the {ordinals[idx]} video is from the {labels[view]}"
            for idx, view in enumerate(active_views)
        ]
        return f"There are two videos: {'; '.join(parts)}."

    def build_vla_user_text(
        self,
        instruction: str,
        head_intrinsic: torch.Tensor,
        n_states: torch.Tensor,
        active_views: list[str],
        chest_intrinsic: torch.Tensor | None = None,
    ) -> str:
        state_slots = self.state_token * int(n_states.item())
        camera_parts = []
        for view in active_views:
            if view == "head":
                camera_parts.append(self.format_intrinsic_part("Head", head_intrinsic))
            elif view == "chest":
                if chest_intrinsic is None:
                    raise ValueError("chest view is active but chest_intrinsic is missing.")
                camera_parts.append(self.format_intrinsic_part("Chest", chest_intrinsic))
            else:
                raise ValueError(f"Unsupported active view: {view}")
        camera_part = " ".join(camera_parts)
        video_order_part = self.format_video_order_part(active_views)
        return (
            f"Task: {instruction}. {video_order_part} {camera_part} "
            f"States: {state_slots}."
        )

    def build_messages(self, sample: dict[str, Any], prompt_only: bool = False) -> list[dict[str, Any]]:
        """Build one chat conversation from a raw sample.

        Contract:
        - VLA samples are converted into a user turn with visual placeholders plus task text, followed by an
          assistant turn made of repeated `<action>` slot tokens.
        - VLM samples are converted into a standard user/assistant question-answer pair.
        - `prompt_only=True` keeps only the user turn so callers can measure the prompt boundary before assistant
          generation starts.
        """
        is_vla = bool(sample["is_vla_data"].item())

        if is_vla:
            active_views = resolve_active_views(sample)
            user_text = self.build_vla_user_text(
                instruction=sample["instruction"],
                head_intrinsic=sample["intrinsic"],
                n_states=sample["n_states"],
                active_views=active_views,
                chest_intrinsic=sample.get("chest_intrinsic"),
            )
            assistant_text = self.action_token * int(sample["n_actions"].item())
            if self.predict_future_frames:
                user_text += " Predict the next action sequence and future frames."
            else:
                user_text += " Predict the next action sequence."
        else:
            user_text = str(sample["question"]).strip()
            assistant_text = str(sample["answer"]).strip()

        messages = [
            {
                "role": "user",
                "content": [*self.build_visual_content(sample), {"type": "text", "text": user_text}],
            }
        ]
        if not prompt_only:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )
        return messages


class Qwen3VLBatchProcessor:
    """Bridge project batches to the native HF Qwen3-VL processor.

    Kwarg routing contract:
    - `processor_init_kwargs` are forwarded to `AutoProcessor.from_pretrained(...)`.
      Loader kwargs such as `trust_remote_code` are consumed by `transformers` at load time, while recognized
      modality kwargs such as `size` continue to the underlying tokenizer/image/video sub-processors.
    - `processor_call_kwargs` are forwarded to `processor(...)` at encode time.
      `transformers` then routes recognized flat kwargs to text/image/video branches and ignores unsupported keys.
    """

    def __init__(
        self,
        model_name_or_path: str,
        processor_init_kwargs: dict[str, Any] | None = None,
        processor_call_kwargs: dict[str, Any] | None = None,
        ignore_index: int = -100,
        padding_side: str = "right",
        state_token: str = "<state>",
        action_token: str = "<action>",
        camera_token: str = "",
        processor: Any = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.processor_init_kwargs = dict(processor_init_kwargs or {})
        self.processor_call_kwargs = dict(processor_call_kwargs or {})
        self.ignore_index = int(ignore_index)
        self.padding_side = padding_side
        self.state_token = state_token
        self.action_token = action_token
        self.camera_token = camera_token
        self.processor = processor if processor is not None else self.init_processor()
        self.tokenizer = self.processor.tokenizer
        special_tokens = [self.state_token, self.action_token]
        if self.camera_token:
            special_tokens.append(self.camera_token)
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.action_token_id = int(self.tokenizer.convert_tokens_to_ids(self.action_token))

        # Assistant header tokens, used by find_answer_start_idx to locate
        # the prompt/answer boundary. Qwen3-VL tokenizes this to exactly 3
        # tokens; anything else means tokenizer drift, fail loudly.
        self.assistant_header_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        assert len(self.assistant_header_ids) == 3, (
            f"Assistant header must tokenize to 3 tokens, got "
            f"{len(self.assistant_header_ids)}: {self.assistant_header_ids}"
        )

    def init_processor(self):
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3VLBatchProcessor requires transformers with Qwen3-VL support."
            ) from exc

        return AutoProcessor.from_pretrained(
            self.model_name_or_path,
            **self.processor_init_kwargs,
        )

    def render_chat_texts(
        self,
        messages_batch: list[list[dict[str, Any]]],
        add_generation_prompt: bool,
    ) -> list[str]:
        rendered = self.processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(rendered, str):
            return [rendered]
        return list(rendered)

    def append_video_entry(
        self,
        videos: list[Any],
        video_metadata: list[dict[str, Any]],
        video: torch.Tensor,
        video_fps: float,
    ) -> None:
        """Send the full video to the processor."""
        videos.append(video)
        video_metadata.append(build_video_metadata(video, video_fps))

    def build_vision_inputs(
        self, batch_samples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build modality inputs. `videos` is flat in placeholder order;
        per sample head video first, then chest (when present)."""
        images: list[Any] = []
        videos: list[Any] = []
        video_metadata: list[dict[str, Any]] = []

        for sample in batch_samples:
            vision_type = sample["vision_type"]
            if vision_type == "image":
                images.append(build_sample_images(sample["images"]))
            elif vision_type == "video":
                fps = float(sample["video_fps"].item())
                for view in resolve_active_views(sample):
                    if view == "head":
                        video = sample["images"]
                    elif view == "chest":
                        if sample.get("chest_images") is None:
                            raise ValueError("chest view is active but chest_images is missing.")
                        video = sample["chest_images"]
                    else:
                        raise ValueError(f"Unsupported active view: {view}")
                    self.append_video_entry(
                        videos, video_metadata,
                        build_sample_video(video), fps,
                    )
            else:
                raise ValueError(f"Unsupported vision_type: {vision_type}")

        processor_inputs: dict[str, Any] = {}
        if images:
            processor_inputs["images"] = images
        if videos:
            processor_inputs["videos"] = videos
            processor_inputs["video_metadata"] = video_metadata
            processor_inputs["do_sample_frames"] = False
        return processor_inputs

    def encode_messages(
        self,
        messages_batch: list[list[dict[str, Any]]],
        batch_samples: list[dict[str, Any]],
        add_generation_prompt: bool,
        return_rendered_texts: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the native Qwen3-VL processor on one batch.

        Contract:
        - Image samples use `vision_type="image"` and `images` shaped `[H, W, C]` or `[T, H, W, C]`.
        - Video samples use `vision_type="video"`, `images` shaped `[T, H, W, C]`, and `video_fps`.
        - Processor kwargs are passed directly from `processor_call_kwargs`.
        - truncation=True is intentional: uniform seq len lets torch.compile reuse one
          kernel cache. Tune `max_vlm_tokens` so truncation hits <1% of samples;
          truncated samples drop the assistant header and silently get VLM loss=0
          (see `find_answer_start_idx`).
        """
        self.processor.tokenizer.padding_side = self.padding_side

        rendered_texts = self.render_chat_texts(
            messages_batch=messages_batch,
            add_generation_prompt=add_generation_prompt,
        )
        vision_inputs = self.build_vision_inputs(batch_samples)
        encoded = self.processor(
            text=rendered_texts,
            **self.processor_call_kwargs,
            **vision_inputs,
        )
        batch = dict(encoded)
        batch.setdefault("pixel_values", None)
        batch.setdefault("image_grid_thw", None)
        batch.setdefault("pixel_values_videos", None)
        batch.setdefault("video_grid_thw", None)
        if "mm_token_type_ids" not in batch:
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        if return_rendered_texts:
            batch["rendered_texts"] = rendered_texts
        return batch

    def build_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer_start_idx: torch.Tensor,
    ) -> torch.Tensor:
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, self.ignore_index)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        labels = labels.masked_fill(positions < answer_start_idx.unsqueeze(1), self.ignore_index)
        return labels

    def find_answer_start_idx(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Position right after `<|im_start|>assistant\\n` for each sample.

        Truncated training samples may not contain the header; for those we
        return the sequence length so downstream label masking treats the
        whole sequence as prompt (no supervised tokens).
        """
        _, L = input_ids.shape
        header = torch.tensor(self.assistant_header_ids, device=input_ids.device)
        H = header.numel()
        assert L >= H, f"Sequence too short ({L}) for assistant header ({H})"

        windows = input_ids.unfold(dimension=1, size=H, step=1)
        matches = (windows == header).all(dim=-1)

        # At most one header per sample. 0 = truncated (warned below);
        # >1 = multi-turn or corrupted data, which would silently mask wrong
        # tokens, so fail loudly.
        match_counts = matches.sum(dim=1)
        assert (match_counts <= 1).all(), (
            f"Each sample has at most 1 assistant header, got counts "
            f"{match_counts.tolist()}"
        )

        truncated = (match_counts == 0).nonzero(as_tuple=True)[0]
        if truncated.numel() > 0:
            log.warning(
                "Assistant header missing in %d/%d samples (likely truncated); "
                "their loss will be masked out.",
                truncated.numel(), input_ids.shape[0],
            )

        # argmax on all-False rows returns 0; remap those to L.
        pos = matches.long().argmax(dim=1) + H
        return torch.where(match_counts == 1, pos, torch.full_like(pos, L))
