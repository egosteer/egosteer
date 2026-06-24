from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.dataset.qwen3_vl_batching import (
    Qwen3VLBatchProcessor,
    Qwen3VLChatFormatter,
    resolve_active_views,
)


class UnifiedVLACollator:
    """Collate raw VLA/VLM samples into one multimodal batch.

    This is the top-level batching contract for the project.

    Input sample contract:
    - VLA samples provide `instruction`, `images`, `intrinsic`, `states`, `actions`, `n_states`, `n_actions`,
      `active_views`, `view_mask`, `vision_type`, `video_fps`, and `is_vla_data=True`.
    - VLM samples provide `question`, `answer`, `images`, `view_mask`, `vision_type`, and `is_vla_data=False`.

    Output contract:
    - The returned batch always includes HF multimodal fields such as `input_ids`, `attention_mask`,
      `pixel_values`/`pixel_values_videos`, grid metadata, `mm_token_type_ids`, and `answer_start_idx`.
    - For VLM samples, `labels` supervise the assistant text after the prompt boundary.
    - For VLA samples, `labels` stay masked because action supervision is handled by slot tokens and action heads.
    """

    def __init__(
        self,
        formatter: Qwen3VLChatFormatter,
        batch_processor: Qwen3VLBatchProcessor,
        mode: str = "train",
        debug_capture_texts: bool = False,
        debug_profile_timing: bool = False,
    ):
        self.formatter = formatter
        self.batch_processor = batch_processor
        self.ignore_index = batch_processor.ignore_index
        self.debug_capture_texts = bool(debug_capture_texts)
        self.debug_profile_timing = bool(debug_profile_timing)
        self.mode = mode
        self.prompt_only_input = "infer" in mode
        self.batch_processor.padding_side = "left" if mode == "infer-ar" else "right"

        if self.formatter.state_token != self.batch_processor.state_token:
            raise ValueError("Formatter and batch processor must share the same state token.")
        if self.formatter.action_token != self.batch_processor.action_token:
            raise ValueError("Formatter and batch processor must share the same action token.")
        if self.batch_processor.camera_token != self.formatter.camera_token:
            raise ValueError("Formatter and batch processor must share the same camera token.")

    def collate_values(self, values: list[Any]) -> Any:
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        return values

    def collate_raw(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one batch from raw dataset samples.

        Single processor pass: full conversation in training, prompt-only +
        add_generation_prompt in inference. Answer boundary recovered from
        input_ids via find_answer_start_idx.
        """
        collate_start = time.perf_counter()

        messages = [self.formatter.build_messages(sample, prompt_only=self.prompt_only_input) for sample in samples]
        main_batch = self.batch_processor.encode_messages(
            messages_batch=messages,
            batch_samples=samples,
            add_generation_prompt=self.prompt_only_input,
            return_rendered_texts=self.debug_capture_texts,
        )

        input_ids = main_batch["input_ids"].to(dtype=torch.long)
        attention_mask = main_batch["attention_mask"].to(dtype=torch.long)
        mm_token_type_ids = main_batch["mm_token_type_ids"].to(dtype=torch.long)
        image_grid_thw = main_batch["image_grid_thw"]
        video_grid_thw = main_batch["video_grid_thw"]

        is_vla_mask = torch.stack([sample["is_vla_data"] for sample in samples]).to(
            device=input_ids.device,
            dtype=torch.bool,
        )
        answer_start_idx = self.batch_processor.find_answer_start_idx(input_ids).to(
            device=input_ids.device, dtype=torch.long,
        )
        labels = torch.full_like(input_ids, self.ignore_index)

        if not self.prompt_only_input:
            vlm_indices = (~is_vla_mask).nonzero(as_tuple=False).flatten().tolist()
            if vlm_indices:
                vlm_answer_start_idx = answer_start_idx[vlm_indices]
                labels[vlm_indices] = self.batch_processor.build_labels(
                    input_ids=input_ids[vlm_indices],
                    attention_mask=attention_mask[vlm_indices],
                    answer_start_idx=vlm_answer_start_idx,
                ).to(device=input_ids.device)

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": main_batch["pixel_values"],
            "image_grid_thw": image_grid_thw.to(dtype=torch.long) if image_grid_thw is not None else None,
            "pixel_values_videos": main_batch["pixel_values_videos"],
            "video_grid_thw": video_grid_thw.to(dtype=torch.long) if video_grid_thw is not None else None,
            "mm_token_type_ids": mm_token_type_ids,
            "answer_start_idx": answer_start_idx,
        }

        collatable_keys = [
            "is_vla_data", "states", "actions", "actions_valid_mask",
            "n_states", "n_actions", "n_future_frames",
            "depth_values", "has_depth_values", "future_frames",
            "intrinsic",
            "chest_intrinsic", "chest_future_frames",
            "future_head_motion", "future_chest_motion",
            "view_mask",
        ]
        for key in collatable_keys:
            if all(key in s for s in samples):
                batch[key] = self.collate_values([s[key] for s in samples])
        if "view_mask" not in batch:
            raise KeyError("All samples must provide view_mask.")

        # Token mode: emit one intrinsic per rendered <camera> slot in
        # sample-major active-view order. VLM samples contribute no rows.
        if self.formatter.camera_intrinsic_mode == "token":
            cam_list = []
            for s in samples:
                if not bool(s["is_vla_data"].item()):
                    continue
                for view in resolve_active_views(s):
                    if view == "head":
                        cam_list.append(s["intrinsic"])
                    elif view == "chest":
                        cam_list.append(s["chest_intrinsic"])
                    else:
                        raise ValueError(f"Unsupported active view: {view}")
            if cam_list:
                batch["camera_intrinsic"] = torch.stack(cam_list)

        if self.debug_capture_texts:
            batch["debug_messages"] = messages
            batch["debug_texts"] = main_batch["rendered_texts"]

        if self.debug_profile_timing:
            sample_profiles = [
                sample.get("debug_sample_profile")
                for sample in samples
                if sample.get("debug_sample_profile") is not None
            ]
            worker_ids = sorted({int(profile["worker_id"]) for profile in sample_profiles})
            sample_to_data_total_s = sum(float(profile["sample_to_data_s"]) for profile in sample_profiles)
            preprocess_total_s = sum(float(profile["preprocess_total_s"]) for profile in sample_profiles)
            profiled_samples = len(sample_profiles)
            collate_total_s = time.perf_counter() - collate_start
            batch["debug_collate_profile"] = {
                "worker_ids": worker_ids,
                "profiled_samples": profiled_samples,
                "collator_s": collate_total_s,
                "sample_to_data_total_s": sample_to_data_total_s,
                "sample_to_data_avg_s": sample_to_data_total_s / profiled_samples if profiled_samples else 0.0,
                "preprocess_total_s": preprocess_total_s,
                "preprocess_avg_s": preprocess_total_s / profiled_samples if profiled_samples else 0.0,
            }

        return batch

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return self.collate_raw(samples)


class ConcatDataCollator:
    """Collator that concatenates samples along the first dimension.

    Used for normalizer fitting where streaming statistics are accumulated
    from variable-length batches.
    """

    def __call__(self, data_list):
        batch = {}
        for key in data_list[0].keys():
            if isinstance(data_list[0][key], torch.Tensor):
                batch[key] = torch.cat([item[key] for item in data_list], dim=0)
            elif isinstance(data_list[0][key], np.ndarray):
                batch[key] = np.concatenate([item[key] for item in data_list], axis=0)
            else:
                batch[key] = [item[key] for item in data_list]
        batch["_batch_num_samples"] = len(data_list)
        return batch
