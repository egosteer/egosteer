"""
WebDataset-based VLA datasets for EgoSteer training and normalizer fitting.
"""

import math
import warnings
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from src.model.common.normalizer import LinearNormalizer
from src.dataset.unified_vla_collator import UnifiedVLACollator
from src.utils.pytorch_util import dict_apply
from .data_transforms import (
    compute_relative_motion_padded,
    process_state_action,
    process_image,
    resize_frames,
)
from .sanity_checks import DataChecker, current_worker_id
from .unified_vla_collator import ConcatDataCollator
from .wds_dataset import (
    build_blended_dataset, build_wds_pipeline, WindowConfig,
    expand_shard_patterns,
)


@dataclass(frozen=True)
class ViewDropoutConfig:
    """Per-sample view dropout (train only). keep_both is the implicit residual."""
    drop_head: float = 0.0
    drop_chest: float = 0.0

    def __post_init__(self) -> None:
        if self.drop_head < 0.0 or self.drop_chest < 0.0:
            raise ValueError(
                f"view_dropout probabilities must be non-negative, "
                f"got drop_head={self.drop_head}, drop_chest={self.drop_chest}"
            )
        if self.drop_head + self.drop_chest > 1.0 + 1e-6:
            raise ValueError(
                f"drop_head + drop_chest must be <= 1.0, "
                f"got {self.drop_head + self.drop_chest:.6f}"
            )


class VLAWdsDataset(torch.utils.data.IterableDataset):
    """WebDataset-backed VLA dataset for EgoSteer training.

    This is an IterableDataset that streams data from WebDataset shards.

    Usage:
        dataset = VLAWdsDataset(
            wds_datasets=[
                {"shard_urls": "example_data/vla/train/shard-*.tar", "weight": 1.0, "name": "example"},
                ...
            ],
            shape_meta=shape_meta,
        )
        dataset.set_collator(collator)
        dataset.set_normalizer(normalizer)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=20)
    """
    def __init__(
        self,
        wds_datasets: List[Dict],
        shape_meta: Dict,
        use_relative_action: bool = False,
        mode: str = "train",
        depth_clip_range=None,
        shuffle_buffer: int = 16384,
        shuffle_initial: Optional[int] = None,
        return_dataset_info: bool = False,
        val_wds_datasets: Optional[List[Dict]] = None,
        video_base_fps: float = 30.0,
        target_image_size: Optional[List[int]] = None,
        debug_capture_raw_sample: bool = False,
        debug_capture_processed_sample: bool = False,
        debug_profile_timing: bool = False,
        load_depth: bool = False,
        load_chest: bool = False,
        view_dropout: ViewDropoutConfig = ViewDropoutConfig(),
        keep_ratio: float = 1.0,
        sanity_checks: Optional[Dict] = None,
        dagger_quality_filter: bool = True,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.motion_type = shape_meta["obs"]["state"]["type"]
        self.hand_ndim = shape_meta["obs"]["state"]["hand"]["shape"][-1] // 2
        self.action_ndim = shape_meta["action"]["shape"][-1]
        self.depth_image_shape = shape_meta["obs"]["depth"]["shape"]
        self.use_relative_action = use_relative_action
        self.mode = mode
        self.depth_clip_range = depth_clip_range
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_initial = shuffle_initial
        self.wds_datasets = wds_datasets
        self.val_wds_datasets = val_wds_datasets
        self.return_dataset_info = return_dataset_info
        self.video_base_fps = float(video_base_fps)
        # Modality flags are forwarded to build_blended_dataset so the
        # WebDataset tar reader filters members at read time.  Skipping
        # depth decoding + augment_depth saves ~25 ms/sample (depth
        # augmentation is the single biggest CPU cost).
        self.load_depth = bool(load_depth)
        self.load_chest = bool(load_chest)
        self.view_dropout = view_dropout
        assert 0.0 < keep_ratio <= 1.0, f"keep_ratio must be in (0, 1], got {keep_ratio}"
        self.keep_ratio = float(keep_ratio)
        self.sanity_checks = dict(sanity_checks or {})
        self.dagger_quality_filter = bool(dagger_quality_filter)
        # (H, W) tuple or None. Resize all RGB frames to this resolution
        # before HF processor. Required when world model is enabled so that
        # temporal attention patches share identical spatial semantics.
        self.target_image_size = tuple(target_image_size) if target_image_size else None
        self.debug_capture_raw_sample = bool(debug_capture_raw_sample)
        self.debug_capture_processed_sample = bool(debug_capture_processed_sample)
        self.debug_profile_timing = bool(debug_profile_timing)

        self.collator = None
        self.normalizer = None

        # Sampling config from shape_meta.
        self.action_horizon = shape_meta["action"]["horizon"]
        self.state_horizon = shape_meta["obs"]["state"]["horizon"]
        self.image_horizon = shape_meta["obs"]["rgb"]["horizon"]

        ff_cfg = shape_meta.get("future_frame", {})
        self.future_frame_horizon = int(ff_cfg.get("horizon", 0))
        self.future_frame_stride = int(ff_cfg.get("stride", 30))

        self.window_config = WindowConfig(
            action_horizon=shape_meta["action"]["horizon"],
            action_stride=shape_meta["action"]["stride"],
            state_horizon=shape_meta["obs"]["state"]["horizon"],
            state_stride=shape_meta["obs"]["state"]["stride"],
            image_horizon=shape_meta["obs"]["rgb"]["horizon"],
            image_stride=shape_meta["obs"]["rgb"]["stride"],
            history_pad_mode=shape_meta.get("history_pad_mode", "repeat"),
            action_pad_mode=shape_meta["action"].get("pad_mode", "truncate"),
            future_frame_horizon=self.future_frame_horizon,
            future_frame_stride=self.future_frame_stride,
            future_frame_pad_mode=ff_cfg.get("pad_mode", "repeat"),
            dagger_quality_filter=self.dagger_quality_filter,
        )

        # process_image only checks truthiness; actual color aug lives in
        # data_transforms.COLOR_AUG (albumentations-based).
        self.aug_transform = (self.mode == "train")

        self.checker = DataChecker(sanity_cfg=self.sanity_checks)

    def sample_active_views(self, *, has_chest: bool) -> list[str]:
        """Sample which views are active for the current sample.

        Train mode + has_chest: roll view_dropout to optionally drop one side.
        Otherwise (val mode, or no chest available): keep all available views.
        view_mask is derived from the result at the data-build site.
        """
        if not has_chest or self.mode != "train":
            return ["head", "chest"] if has_chest else ["head"]

        drop_head = self.view_dropout.drop_head
        drop_chest = self.view_dropout.drop_chest
        keep_both = 1.0 - drop_head - drop_chest
        choice = np.random.choice(
            ["keep_both", "drop_head", "drop_chest"],
            p=[keep_both, drop_head, drop_chest],
        )
        if choice == "drop_head":
            return ["chest"]
        if choice == "drop_chest":
            return ["head"]
        return ["head", "chest"]

    def set_collator(self, collator):
        """Set the batch collator used to build model inputs."""
        self.collator = collator

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set the normalizer for state/action."""
        self.normalizer = normalizer

    def build_raw_model_inputs(
        self,
        instruction,
        image,
        intrinsic,
        active_views,
        chest_image=None,
        chest_intrinsic=None,
    ):
        view_mask = np.array(
            ["head" in active_views, "chest" in active_views], dtype=bool
        )
        data = {
            "images": image,
            "instruction": instruction,
            "intrinsic": intrinsic,
            "active_views": active_views,
            "view_mask": view_mask,
            "vision_type": "video",
            "video_fps": np.array(
                self.video_base_fps / self.window_config.image_stride,
                dtype=np.float32,
            ),
            "has_depth_values": np.array(False, dtype=bool),
        }
        if chest_image is not None:
            data["chest_images"] = chest_image
            data["chest_intrinsic"] = chest_intrinsic
        return data

    def copy_debug_value(self, value):
        """Create a detached debug copy of one sample field."""
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: self.copy_debug_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.copy_debug_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.copy_debug_value(item) for item in value)
        return value

    def build_debug_raw_sample(self, sample):
        """Keep the dataset sample before state/image processing for offline inspection."""
        debug_sample = {
            "wrist_state": sample["wrist_state"].astype(np.float32),
            "hand_state": sample["hand_state"].astype(np.float32),
            "wrist_action": sample["wrist_action"].astype(np.float32),
            "hand_action": sample["hand_action"].astype(np.float32),
            "extrinsic": sample["extrinsic"].astype(np.float32).reshape(4, 4),
            "intrinsic": sample["intrinsic"].astype(np.float32),
            "instruction": self.copy_debug_value(sample["instruction"]),
            "instruction_num": np.array(sample["instruction_num"], dtype=np.int32),
            "presence": np.array(sample.get("presence", 3), dtype=np.int32),
            "image": sample["image"].copy(),
        }
        if "dataset_name" in sample:
            debug_sample["dataset_name"] = self.copy_debug_value(sample["dataset_name"])
        if "episode_index" in sample:
            debug_sample["episode_index"] = np.array(sample["episode_index"], dtype=np.int32)
        return debug_sample

    def build_debug_processed_sample(self, data):
        """Keep the exact collator input sample for offline inspection."""
        return {
            key: self.copy_debug_value(value)
            for key, value in data.items()
            if not key.startswith("debug_")
        }

    def sample_to_data(self, sample):
        """Convert one WebDataset sample into the raw VLA sample schema used by the project.

        This is the dataset-side producer contract for `UnifiedVLACollator`.
        The returned mapping contains visual history, instruction text, intrinsic parameters, padded state/action
        tensors, action-valid masks, and bookkeeping fields such as `n_states`, `n_actions`, and `is_vla_data`.
        """
        self.checker.check(sample_schema=(sample, {
            "required_keys": (
                "wrist_state", "hand_state", "wrist_action", "hand_action",
                "extrinsic", "intrinsic", "instruction", "instruction_num", "image",
            ),
            "expected_last_dim": {
                "wrist_state": 18,
                "hand_state": 30,
                "wrist_action": 18,
                "hand_action": 30,
                "extrinsic": 16,
                "intrinsic": 4,
            },
        }))

        # Cheap structural checks first so bad samples skip JPEG decode + transforms.
        intrinsic_raw = sample["intrinsic"].astype(np.float32)
        extrinsic_raw = sample["extrinsic"].astype(np.float32).reshape(4, 4)
        self.checker.check(
            intrinsic=intrinsic_raw,
            extrinsic=extrinsic_raw,
            instruction=(sample["instruction"], sample["instruction_num"]),
        )
        if sample.get("chest_image") is not None:
            self.checker.check(
                intrinsic=sample["chest_intrinsic"].astype(np.float32),
                extrinsic=sample["chest_extrinsic"].astype(np.float32).reshape(4, 4),
            )
        future_head_ext_raw = sample.get("future_head_extrinsic")
        if self.future_frame_horizon > 0 and future_head_ext_raw is not None:
            self.checker.check(
                extrinsic=future_head_ext_raw.astype(np.float32).reshape(-1, 4, 4),
            )

        # Pre-check raw lowdim before process_state_action: the downstream
        # transform_hand_points_to_wrist_frame -> torch.linalg.pinv path
        # crashes (LinAlgError) when wrist_state contains non-finite values
        # rather than producing a finite-but-bad output the post-check could
        # catch. Hoisting the finite check upstream lets DataSkipError fire
        # cleanly on shards with corrupt poses (e.g. shards with
        # invalid rot6d rotations).
        wrist_state = sample["wrist_state"].astype(np.float32)
        hand_state = sample["hand_state"].astype(np.float32)
        wrist_action = sample["wrist_action"].astype(np.float32)
        hand_action = sample["hand_action"].astype(np.float32)
        self.checker.check(finite={
            "wrist_state": wrist_state,
            "hand_state": hand_state,
            "wrist_action": wrist_action,
            "hand_action": hand_action,
        })
        self.checker.check(
            rot6d={"wrist_state": wrist_state, "wrist_action": wrist_action},
            state_action_delta=(wrist_state, hand_state, wrist_action, hand_action),
        )

        state, action = process_state_action(
            wrist_state=wrist_state,
            hand_state=hand_state,
            wrist_action=wrist_action,
            hand_action=hand_action,
            extrinsic=extrinsic_raw,
            normalizer=self.normalizer,
            hand_ndim=self.hand_ndim,
            motion_type=self.motion_type,
            use_relative_action=self.use_relative_action,
        )
        self.checker.check(
            finite={"state": state, "action": action},
            rot6d={"state": state, "action": action},
        )

        image, depth_images, intrinsic = process_image(
            sample["image"],
            sample.get("depth", None),
            intrinsic_raw,
            self.aug_transform,
            self.depth_clip_range,
            target_size=self.target_image_size,
        )
        self.checker.check(
            image=image,
            depth=(depth_images, self.depth_clip_range),
            finite={"image": image, "depth_images": depth_images},
        )

        chest_image = None
        chest_intrinsic = None
        if sample.get("chest_image") is not None:
            chest_intrinsic = sample["chest_intrinsic"].astype(np.float32)
            chest_image, _, chest_intrinsic = process_image(
                sample["chest_image"],
                sample.get("chest_depth", None),
                chest_intrinsic,
                self.aug_transform,
                self.depth_clip_range,
                target_size=self.target_image_size,
            )
            self.checker.check(image=chest_image, finite={"chest_image": chest_image})
        elif self.load_chest:
            # load_chest=True means the chest view is required. A missing
            # chest_image here typically means legacy 'breast'-named data, which
            # the chest-only loader would otherwise silently degrade to
            # head-only. Raise a plain ValueError (NOT a DataSkipError) so
            # attach_sample_ctx re-raises it as a fatal RuntimeError with the
            # sample locator attached, instead of dropping the sample.
            raise ValueError(
                "load_chest=True but sample has no 'chest_image': legacy "
                "'breast'-named data is unreadable by the chest-only loader. "
                "Rename breast->chest in the data, or set load_chest=False."
            )
        # Sample dropout perspective
        active_views = self.sample_active_views(has_chest=chest_image is not None)
        instruction = sample["instruction"]
        instruction_num = sample["instruction_num"]

        # Sample a random instruction from candidates
        if self.mode == "train":
            idx = np.random.randint(0, instruction_num)
        else:
            idx = 0
        # instruction may be a single string or list
        if isinstance(instruction, list):
            instruction = instruction[idx]

        state_pad = np.zeros(
            (self.state_horizon, *state.shape[1:]), dtype=np.float32)
        state_pad[:state.shape[0]] = state
        action_pad = np.zeros(
            (self.action_horizon, *action.shape[1:]), dtype=np.float32)
        actions_valid_mask = np.zeros(
            (self.action_horizon, *action.shape[1:]), dtype=bool)
        action_pad[:action.shape[0]] = action
        actions_valid_mask[:action.shape[0]] = True

        data = self.build_raw_model_inputs(
            instruction=instruction,
            image=image,
            intrinsic=intrinsic,
            active_views=active_views,
            chest_image=chest_image,
            chest_intrinsic=chest_intrinsic,
        )

        data.update({
            "states": state_pad,
            "n_states": np.array(state.shape[0], dtype=np.int32),
            "actions": action_pad,
            "actions_valid_mask": actions_valid_mask,
            "n_actions": np.array(action.shape[0], dtype=np.int32),
            "is_vla_data": np.array(True, dtype=bool),
        })

        # Future frames for world model supervision (raw uint8, no augmentation).
        # Always emit future_frames when K > 0 so collator can stack; chest
        # shares head's valid length (same frame_refs window).
        K = self.future_frame_horizon
        if K > 0:
            if self.target_image_size is None:
                raise ValueError(
                    "target_image_size must be set when future_frame_horizon > 0. "
                    "Set data.target_image_size in the config."
                )
            tH, tW = self.target_image_size

            def pad_future(source):
                # Valid count = len(source): gather_future_refs already respected
                # future_frame_pad_mode (repeat fills to K, truncate yields real count).
                frames = np.zeros((K, tH, tW, 3), dtype=np.uint8)
                if source is None:
                    return frames, 0
                n = min(source.shape[0], K)
                if n > 0:
                    frames[:n] = resize_frames(source[:n], self.target_image_size)
                return frames, n

            ff, n_valid = pad_future(sample.get("future_frames"))
            data["future_frames"] = ff
            data["n_future_frames"] = np.array(n_valid, dtype=np.int32)

            # Relative head-camera motion = inv(T_current) @ T_future[k], i.e.
            # future pose expressed in the current camera frame. Invalid steps
            # (>= n_valid) are zero-filled; downstream mask drops them.
            head_motion = compute_relative_motion_padded(
                current_flat16=sample["extrinsic"],
                future_flat=sample.get("future_head_extrinsic"),
                n_valid=n_valid, K=K,
            )
            data["future_head_motion"] = head_motion

            chest_ff, _ = pad_future(sample.get("chest_future_frames"))  # zero-filled when chest RGB is absent
            data["chest_future_frames"] = chest_ff
            # Gate on the extrinsics actually read below, not on chest RGB frames:
            # the two come from independent sources (load_chest vs meta["cameras"]).
            if sample.get("future_chest_extrinsic") is not None:
                chest_motion = compute_relative_motion_padded(
                    current_flat16=sample.get("chest_extrinsic"),
                    future_flat=sample.get("future_chest_extrinsic"),
                    n_valid=n_valid, K=K,
                )
                data["future_chest_motion"] = chest_motion
            else:
                data["future_chest_motion"] = np.zeros((K, 16), dtype=np.float32)
        else:
            data["n_future_frames"] = np.array(0, dtype=np.int32)
        if self.return_dataset_info:
            data["dataset_name"] = sample["dataset_name"]
            data["episode_index"] = sample["episode_index"]
        self.checker.check(finite=data, post_normalize=data)

        if self.debug_capture_raw_sample:
            data["debug_raw_sample"] = self.build_debug_raw_sample(sample)
        if self.debug_capture_processed_sample:
            data["debug_processed_sample"] = self.build_debug_processed_sample(data)
        return data

    def build_pipeline(self):
        """Build the WebDataset pipeline.

        Training: resampled infinite stream with shuffle.
        Validation: finite single-pass, no shuffle.
        """
        datasets_config = []
        for ds in self.wds_datasets:
            datasets_config.append({
                "shard_urls": ds["shard_urls"],
                "weight": ds.get("weight", 1.0),
                "name": ds.get("name", "unknown"),
            })

        def preprocess_fn(sample):
            # DataSkipError -> attach_sample_ctx logs and drops the sample;
            # any other failure -> attach_sample_ctx re-raises with locator.
            self.checker.note_sample_seen()
            preprocess_start = time.perf_counter()
            sample_to_data_start = time.perf_counter()
            data = self.sample_to_data(sample)
            sample_to_data_s = time.perf_counter() - sample_to_data_start
            torch_data = dict_apply(
                data,
                lambda x: torch.from_numpy(x)
                if isinstance(x, np.ndarray) else x,
            )
            preprocess_total_s = time.perf_counter() - preprocess_start
            if self.debug_profile_timing:
                torch_data["debug_sample_profile"] = {
                    "worker_id": current_worker_id(),
                    "sample_to_data_s": sample_to_data_s,
                    "preprocess_total_s": preprocess_total_s,
                }
            return torch_data

        def strip_key(src):
            # wds .map() may auto-inject __key__; collation expects it gone.
            for sample in src:
                sample.pop("__key__", None)
                yield sample

        pipeline = build_blended_dataset(
            datasets_config=datasets_config,
            config=self.window_config,
            load_image=True,
            load_depth=self.load_depth,
            load_chest=self.load_chest,
            preprocess_fn=preprocess_fn,
            shuffle_buffer=self.shuffle_buffer,
            shuffle_initial=self.shuffle_initial,
            mode=self.mode,
            keep_ratio=self.keep_ratio,
            checker=self.checker,
        )
        return strip_key(pipeline)

    def __iter__(self):
        pipeline = self.build_pipeline()
        return iter(pipeline)

    def get_validation_dataset(self):
        """Create a validation dataset from separate val shard URLs.
        """
        assert self.val_wds_datasets is not None, "val_wds_datasets is not set"
        val_dataset = VLAWdsDataset(
            wds_datasets=self.val_wds_datasets,
            shape_meta=self.shape_meta,
            use_relative_action=self.use_relative_action,
            mode="val",
            depth_clip_range=self.depth_clip_range,
            shuffle_buffer=0,
            return_dataset_info=self.return_dataset_info,
            target_image_size=list(self.target_image_size) if self.target_image_size else None,
            debug_capture_raw_sample=self.debug_capture_raw_sample,
            debug_capture_processed_sample=self.debug_capture_processed_sample,
            debug_profile_timing=self.debug_profile_timing,
            load_depth=self.load_depth,
            load_chest=self.load_chest,
            keep_ratio=1.0,
            sanity_checks=self.sanity_checks,
            dagger_quality_filter=self.dagger_quality_filter,
        )
        if self.collator is not None:
            val_dataset.set_collator(self.collator)
        if self.normalizer is not None:
            val_dataset.set_normalizer(self.normalizer)
        return val_dataset

    def get_collator(self):
        """Build a collator copy configured for this dataset's mode."""
        assert self.collator is not None, "Collator not set"
        return UnifiedVLACollator(
            formatter=self.collator.formatter,
            batch_processor=deepcopy(self.collator.batch_processor),
            mode=self.mode,
            debug_capture_texts=self.collator.debug_capture_texts,
            debug_profile_timing=self.collator.debug_profile_timing,
        )


class UnifiedWdsDataset(torch.utils.data.IterableDataset):
    """Unified dataset combining WebDataset VLA with streaming VLM dataset.

    Training mode: VLM samples interleaved at a fixed ratio, VLM auto-restarts.
    Validation mode: all VLA samples first, then all VLM samples sequentially.
    """

    def __init__(
        self,
        vla_dataset: VLAWdsDataset,
        vlm_dataset=None,
        vla_ratio: float = 5 / 6,
        batch_size: int = 20,
        mode: str = "train",
    ):
        super().__init__()
        self.vla_dataset = vla_dataset
        self.vlm_dataset = vlm_dataset
        self.vla_ratio = vla_ratio
        self.batch_size = batch_size
        self.mode = mode
        assert vla_ratio > 0 and vla_ratio <= 1, "vla_ratio must be in (0, 1]"

        self.build_vla_shape_meta()

    def build_vla_shape_meta(self):
        """Build the shape meta for the VLA dataset."""
        chunk_config = self.vla_dataset.window_config
        action_ndim = self.vla_dataset.action_ndim
        self.shape_meta = {
            "states": (chunk_config.state_horizon, action_ndim),
            "actions": (chunk_config.action_horizon, action_ndim),
            "n_states": 1,
            "n_actions": 1,
            # (T, 3, H, W)
            "depth_values": (chunk_config.image_horizon, 3, *self.vla_dataset.depth_image_shape),
            "has_depth_values": 1,
        }

    def get_collator(self):
        return self.vla_dataset.get_collator()

    def get_validation_dataset(self):
        """Create a unified validation dataset.
        """
        vla_val = self.vla_dataset.get_validation_dataset()

        vlm_val = None
        has_vlm_val = getattr(self.vlm_dataset, "val_wds_datasets", None) is not None
        if (
            self.vlm_dataset is not None
            and has_vlm_val
            and hasattr(self.vlm_dataset, 'get_validation_dataset')
        ):
            vlm_val = self.vlm_dataset.get_validation_dataset()

        return UnifiedWdsDataset(
            vla_dataset=vla_val,
            vlm_dataset=vlm_val,
            mode="val",
        )

    def __iter__(self):
        if self.mode == 'train':
            yield from self.iter_train()
        else:
            yield from self.iter_val()

    def iter_train(self):
        """Interleave VLA and VLM at the configured ratio."""
        vla_iter = iter(self.vla_dataset)

        if self.vlm_dataset is None:
            yield from vla_iter
            return

        vlm_iter = iter(self.vlm_dataset)
        vla_per_batch = math.ceil(self.vla_ratio * self.batch_size)
        vlm_per_batch = self.batch_size - vla_per_batch

        count = 0
        for vla_sample in vla_iter:
            yield vla_sample
            count += 1

            if count % vla_per_batch == 0:
                for _ in range(vlm_per_batch):
                    try:
                        vlm_sample = next(vlm_iter)
                    except StopIteration:
                        vlm_iter = iter(self.vlm_dataset)
                        vlm_sample = next(vlm_iter)
                    self.pad_vlm_sample(vlm_sample)
                    yield vlm_sample

    def iter_val(self):
        """Sequential single-pass: all VLA samples, then all VLM samples."""
        for vla_sample in self.vla_dataset:
            yield vla_sample

        if self.vlm_dataset is None:
            return

        for vlm_sample in self.vlm_dataset:
            self.pad_vlm_sample(vlm_sample)
            yield vlm_sample

    def pad_vlm_sample(self, vlm_sample):
        """Pad missing VLA fields on a VLM sample so the collator sees uniform keys."""
        shape_meta = self.shape_meta
        vlm_sample["states"] = torch.zeros(*shape_meta["states"])
        vlm_sample["actions"] = torch.zeros(*shape_meta["actions"])
        vlm_sample["actions_valid_mask"] = torch.zeros(*shape_meta["actions"], dtype=torch.bool)
        vlm_sample["n_states"] = torch.tensor(0, dtype=torch.int32)
        vlm_sample["n_actions"] = torch.tensor(0, dtype=torch.int32)
        vlm_sample["depth_values"] = torch.zeros(*shape_meta["depth_values"])
        vlm_sample["has_depth_values"] = torch.tensor(False, dtype=torch.bool)
        if "intrinsic" not in vlm_sample:
            vlm_sample["intrinsic"] = torch.zeros(4, dtype=torch.float32)
        if vlm_sample.get("vision_type") == "video":
            vlm_sample["active_views"] = ["head"]
        vlm_sample["view_mask"] = torch.tensor([False, False], dtype=torch.bool)
        vlm_sample["n_future_frames"] = torch.tensor(0, dtype=torch.int32)
        ff_horizon = self.vla_dataset.future_frame_horizon
        if ff_horizon > 0:
            tgt = self.vla_dataset.target_image_size
            if tgt is None:
                raise ValueError(
                    "target_image_size must be set when future_frame_horizon > 0. "
                    "Set data.target_image_size in the config."
                )
            tH, tW = tgt
            vlm_sample["future_frames"] = torch.zeros(ff_horizon, tH, tW, 3, dtype=torch.uint8)
            vlm_sample["future_head_motion"] = torch.zeros(ff_horizon, 16, dtype=torch.float32)
            vlm_sample["chest_future_frames"] = torch.zeros(ff_horizon, tH, tW, 3, dtype=torch.uint8)
            vlm_sample["future_chest_motion"] = torch.zeros(ff_horizon, 16, dtype=torch.float32)
        if self.vla_dataset.load_chest:
            vlm_sample["chest_intrinsic"] = torch.zeros(4, dtype=torch.float32)
        if getattr(self.vla_dataset, "debug_capture_raw_sample", False):
            vlm_sample["debug_raw_sample"] = None
        if getattr(self.vla_dataset, "debug_capture_processed_sample", False):
            vlm_sample["debug_processed_sample"] = None
        if getattr(self.vla_dataset, "debug_profile_timing", False):
            vlm_sample["debug_sample_profile"] = None


class VLALowLevelWdsDataset(torch.utils.data.IterableDataset):
    """Low-level WebDataset for normalizer fitting (state/action only, no images).

    This dataset is dedicated to normalizer computation. In practice it should
    run with ``mode="val"`` so the scan is a finite single pass over shards,
    without the training-time ``resampled=True`` behavior.

    Sampling notes:
    - ``max_total_shards`` caps the total number of scanned shards. This keeps
      the approximation logic at shard level instead of introducing frame-level
      early stop semantics inside the dataloader.
    - ``min_shards_per_dataset`` is a per-dataset coverage floor. After the
      floor is reserved, the remaining shard pool is sampled without replacement.
      Larger datasets naturally contribute more shards because they occupy more
      entries in that remaining pool.
    - The final shard list preserves dataset-grouped ordering after shard
      selection is finished. Since selected shards are scanned to completion,
      the normalizer statistics depend on coverage rather than on an additional
      mixing order.
    - The collator intentionally concatenates rows instead of stacking windows.
      History/action horizons can be shorter under ``truncate`` mode; concat
      preserves only valid rows and lets the streaming normalizer consume a flat
      matrix directly.
    """

    def __init__(
        self,
        wds_datasets: List[Dict],
        shape_meta: Dict,
        use_relative_action: bool = False,
        mode: str = "val",
        max_total_shards: Optional[int] = None,
        min_shards_per_dataset: int = 8,
        seed: int = 0,
        sanity_checks: Optional[Dict] = None,
        dagger_quality_filter: bool = True,
    ):
        super().__init__()
        self.wds_datasets = wds_datasets
        self.shape_meta = shape_meta
        self.motion_type = shape_meta["obs"]["state"]["type"]
        self.hand_ndim = shape_meta["obs"]["state"]["hand"]["shape"][-1] // 2
        self.use_relative_action = use_relative_action
        self.mode = mode
        self.max_total_shards = max_total_shards
        self.min_shards_per_dataset = min_shards_per_dataset
        self.seed = seed
        self.sanity_checks = dict(sanity_checks or {})
        self.dagger_quality_filter = bool(dagger_quality_filter)

        if self.mode != "val":
            warnings.warn(
                "VLALowLevelWdsDataset is intended for normalizer fitting and usually "
                "should run with mode='val' for a finite single-pass scan."
            )
        if self.min_shards_per_dataset < 1:
            raise ValueError("min_shards_per_dataset must be >= 1")
        if self.max_total_shards is not None and self.max_total_shards < 1:
            raise ValueError("max_total_shards must be >= 1 when provided")

        # Normalizer fitting only reads lowdim; no future-frame supervision.
        self.window_config = WindowConfig(
            action_horizon=shape_meta["action"]["horizon"],
            action_stride=shape_meta["action"]["stride"],
            state_horizon=shape_meta["obs"]["state"]["horizon"],
            state_stride=shape_meta["obs"]["state"]["stride"],
            image_horizon=shape_meta["obs"]["rgb"]["horizon"],
            image_stride=shape_meta["obs"]["rgb"]["stride"],
            history_pad_mode=shape_meta.get("history_pad_mode", "repeat"),
            action_pad_mode=shape_meta["action"].get("pad_mode", "truncate"),
            dagger_quality_filter=self.dagger_quality_filter,
        )

        self.checker = DataChecker(sanity_cfg=self.sanity_checks)

    def sample_to_data(self, sample):
        """Extract lowdim fields and compute state/action."""
        self.checker.check(sample_schema=(sample, {
            "required_keys": ("wrist_state", "hand_state", "wrist_action", "hand_action", "extrinsic"),
            "expected_last_dim": {
                "wrist_state": 18,
                "hand_state": 30,
                "wrist_action": 18,
                "hand_action": 30,
                "extrinsic": 16,
            },
        }))

        extrinsic = sample["extrinsic"].astype(np.float32).reshape(4, 4)
        self.checker.check(extrinsic=extrinsic)

        # Same hoisted finite check as VLAWdsDataset.sample_to_data: catch
        # NaN/Inf in raw lowdim before process_state_action's pinv blows up.
        wrist_state = sample["wrist_state"].astype(np.float32)
        hand_state = sample["hand_state"].astype(np.float32)
        wrist_action = sample["wrist_action"].astype(np.float32)
        hand_action = sample["hand_action"].astype(np.float32)
        self.checker.check(finite={
            "wrist_state": wrist_state,
            "hand_state": hand_state,
            "wrist_action": wrist_action,
            "hand_action": hand_action,
        })
        self.checker.check(
            rot6d={"wrist_state": wrist_state, "wrist_action": wrist_action},
            state_action_delta=(wrist_state, hand_state, wrist_action, hand_action),
        )

        state, action = process_state_action(
            wrist_state=wrist_state,
            hand_state=hand_state,
            wrist_action=wrist_action,
            hand_action=hand_action,
            extrinsic=extrinsic,
            normalizer=None,
            hand_ndim=self.hand_ndim,
            motion_type=self.motion_type,
            use_relative_action=self.use_relative_action,
        )
        self.checker.check(
            finite={"state": state, "action": action},
            rot6d={"state": state, "action": action},
        )

        if not self.use_relative_action:
            return {
                "motions": np.concatenate([state, action], axis=0),
            }
        return {
            "states": state,
            "actions": action,
        }

    def build_shard_groups(self):
        """Expand shard globs and shuffle each dataset independently."""
        shard_groups = []
        for dataset_index, dataset_cfg in enumerate(self.wds_datasets):
            shard_patterns = dataset_cfg["shard_urls"]
            shard_urls, shard_patterns_metadata = expand_shard_patterns(shard_patterns)
            if not shard_urls:
                warnings.warn(
                    f"No shards found for {dataset_cfg.get('name', '?')}: "
                    f"{shard_patterns_metadata}, skipping."
                )
                continue

            rng = np.random.default_rng(self.seed + dataset_index)
            order = rng.permutation(len(shard_urls)).tolist()
            shuffled_urls = [shard_urls[idx] for idx in order]
            shard_groups.append({
                "dataset_index": dataset_index,
                "name": dataset_cfg.get("name", f"dataset_{dataset_index}"),
                "shard_patterns": shard_patterns_metadata,
                "shard_urls": shuffled_urls,
            })

        if not shard_groups:
            raise ValueError("No shards found across all datasets.")
        return shard_groups

    def select_shards(self):
        """Select final shard URLs with a coverage floor and random remainder sampling."""
        shard_groups = self.build_shard_groups()
        selected_shards = []
        remaining_shards = []
        minimum_selected = 0

        for group in shard_groups:
            base_count = min(len(group["shard_urls"]), self.min_shards_per_dataset)
            selected_shards.extend(
                (group["dataset_index"], shard_url)
                for shard_url in group["shard_urls"][:base_count]
            )
            remaining_shards.extend(
                (group["dataset_index"], shard_url)
                for shard_url in group["shard_urls"][base_count:]
            )
            minimum_selected += base_count

        if self.max_total_shards is not None and self.max_total_shards < minimum_selected:
            raise ValueError(
                "max_total_shards is smaller than the required minimum shard coverage"
            )

        if self.max_total_shards is None:
            extra_budget = len(remaining_shards)
        else:
            extra_budget = min(
                len(remaining_shards),
                self.max_total_shards - minimum_selected,
            )

        if extra_budget > 0:
            rng = np.random.default_rng(self.seed)
            chosen_indices = np.sort(rng.permutation(len(remaining_shards))[:extra_budget])
            selected_shards.extend(
                remaining_shards[int(pool_index)] for pool_index in chosen_indices
            )

        return shard_groups, selected_shards

    def build_shard_urls(self):
        """Build the final shard list used for normalizer fitting."""
        _, selected_shards = self.select_shards()
        return [shard_url for _, shard_url in selected_shards]

    def describe_shard_selection(self):
        """Return a JSON-serializable summary of shard coverage."""
        shard_groups, selected_shards = self.select_shards()
        selected_counts = Counter(dataset_index for dataset_index, _ in selected_shards)
        datasets = []
        for group in shard_groups:
            available_count = len(group["shard_urls"])
            selected_count = selected_counts.get(group["dataset_index"], 0)
            datasets.append({
                "name": group["name"],
                "shard_patterns": group["shard_patterns"],
                "available_shards": available_count,
                "selected_shards": selected_count,
                "full_coverage": selected_count == available_count,
                "selected_fraction": (
                    float(selected_count) / float(available_count)
                    if available_count > 0 else 0.0
                ),
            })

        available_total = sum(item["available_shards"] for item in datasets)
        selected_total = len(selected_shards)
        return {
            "mode": self.mode,
            "seed": self.seed,
            "max_total_shards": self.max_total_shards,
            "min_shards_per_dataset": self.min_shards_per_dataset,
            "available_shards_total": available_total,
            "selected_shards_total": selected_total,
            "full_dataset_coverage": selected_total == available_total,
            "datasets": datasets,
        }

    def build_pipeline(self):
        """Build a streaming pipeline for lowdim-only data."""
        shard_urls = self.build_shard_urls()

        def preprocess_fn(sample):
            self.checker.note_sample_seen()
            data = self.sample_to_data(sample)
            return {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in data.items()
            }

        def strip_key(src):
            for sample in src:
                sample.pop("__key__", None)
                yield sample

        pipeline = build_wds_pipeline(
            shard_urls=shard_urls,
            config=self.window_config,
            # Normalizer fitting only touches base lowdim [0:96]; skip image
            # and depth entirely at tar-read time.
            load_image=False,
            load_depth=False,
            load_chest=False,
            preprocess_fn=preprocess_fn,
            shuffle_buffer=0,
            mode=self.mode,
            use_sliding_window=True,
            checker=self.checker,
        )
        return strip_key(pipeline)

    def __iter__(self):
        pipeline = self.build_pipeline()
        return iter(pipeline)

    def get_collator(self):
        """Return ConcatDataCollator for normalizer fitting.

        Concatenation keeps only valid lowdim rows when truncate padding is used,
        and avoids stacking windows only to flatten them again for streaming stats.
        """
        return ConcatDataCollator()
