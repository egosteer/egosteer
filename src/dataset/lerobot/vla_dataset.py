"""LeRobot episode stream with EgoSteer's existing BC sample semantics."""

from copy import deepcopy

import numpy as np
import torch

from src.model.common.normalizer import LinearNormalizer
from ..data_transforms import (
    ViewDropoutConfig,
    compute_relative_motion_padded,
    process_state_action,
    process_image,
    resize_frames,
)

from .lerobot_dataset import (
    LeRobotEpisodeReader,
    LeRobotDataset,
    WindowConfig,
    camera_parameters,
    read_motion,
)
from ..unified_vla_collator import ConcatDataCollator
from ..unified_dataset import UnifiedDataset


# Keep causal history intact, including model-executed frames before human intervention.
def history_indices(k, horizon, stride, mode):
    ids = [k - i * stride for i in range(horizon - 1, -1, -1)]
    return [max(i, 0) for i in ids] if mode == "repeat" else [i for i in ids if i >= 0]


# Cut supervision at the first low-quality sampled position; episode-tail padding stays separate.
def future_indices(indices, length, mode, quality=None, quality_offset=0):
    refs = []
    for i in indices:
        if i < length:
            if quality is not None and quality[i + quality_offset] == 0:
                break
            refs.append(i)
        elif mode == "repeat":
            refs.append(length - 1)
        else:
            break
    return refs


class VLALeRobotDataset(LeRobotDataset):
    """Sequential LeRobot windows with EgoSteer's shared transforms and collator.

    Workers own disjoint episodes and shuffle compressed windows. Action targets
    are the recorded action column, so every frame can anchor a window. Train
    and val use the episode splits in the same root's info.json.
    """

    lowdim_only = False

    # Translate shape_meta into window rules shared by reading, padding and model inputs.
    def __init__(
        self,
        root,
        shape_meta,
        split="train",
        val_split="val",
        seed=0,
        drop_ratio=0.8,
        shuffle_buffer=16384,
        shuffle_initial=4096,
        reader_kwargs=None,
        use_relative_action=False,
        mode="train",
        depth_clip_range=None,
        return_dataset_info=False,
        video_base_fps=30.0,
        target_image_size=None,
        debug_capture_raw_sample=False,
        debug_capture_processed_sample=False,
        debug_profile_timing=False,
        load_depth=False,
        load_chest=False,
        view_dropout=ViewDropoutConfig(),
        val_stride=1,
        sanity_checks=None,
        resume_warmup_samples=4096,
        dagger_quality_filter=True,
    ):
        super().__init__(
            root,
            split=split,
            val_split=val_split,
            seed=seed,
            mode=mode,
            drop_ratio=drop_ratio,
            shuffle_buffer=shuffle_buffer,
            shuffle_initial=shuffle_initial,
            resume_warmup_samples=resume_warmup_samples,
            val_stride=val_stride,
            reader_kwargs=reader_kwargs,
            target_image_size=target_image_size,
            sanity_checks=sanity_checks,
            return_dataset_info=return_dataset_info,
        )
        self.shape_meta = shape_meta
        self.motion_type = shape_meta["obs"]["state"]["type"]
        self.hand_ndim = shape_meta["obs"]["state"]["hand"]["shape"][-1] // 2
        self.action_ndim = shape_meta["action"]["shape"][-1]
        self.depth_image_shape = shape_meta["obs"]["depth"]["shape"]
        self.use_relative_action = use_relative_action
        self.depth_clip_range = depth_clip_range
        self.video_base_fps = float(video_base_fps)
        self.load_depth = bool(load_depth)
        self.load_chest = bool(load_chest)
        self.view_dropout = view_dropout
        self.debug_capture_raw_sample = bool(debug_capture_raw_sample)
        self.debug_capture_processed_sample = bool(debug_capture_processed_sample)
        self.debug_profile_timing = bool(debug_profile_timing)

        self.normalizer = None
        self.dagger_quality_filter = bool(dagger_quality_filter)

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
        )

        self.aug_transform = self.mode == "train"

        self._validate_config()
        if any(not np.isclose(reader.fps, self.video_base_fps) for reader in self.readers):
            raise ValueError("video_base_fps must match info.json fps")
        if self.mode == "train" and not len(self):
            raise ValueError("training split has no usable anchors")

    def _validate_config(self):
        cfg = self.window_config
        windows = (
            (cfg.action_horizon, cfg.action_stride),
            (cfg.state_horizon, cfg.state_stride),
            (cfg.image_horizon, cfg.image_stride),
        )
        if any(horizon < 1 or stride < 1 for horizon, stride in windows):
            raise ValueError("state/image/action horizons and strides must be positive")
        if cfg.future_frame_horizon < 0 or cfg.future_frame_stride < 1:
            raise ValueError("future horizon must be nonnegative and stride positive")

    def __len__(self):
        """Number of anchors before drop/val_stride; train iteration is infinite."""
        return sum(self.num_anchors(e) for e in range(len(self.episodes)))

    def num_anchors(self, e):
        return int(self.episodes[e]["length"])

    # Assemble state history and recorded action targets from one episode.
    # DAgger checks quality on each target frame's own row.
    def read_window(self, e, k, load_media=True, compressed=False):
        """Build the raw wrist/hand window consumed by sample_to_data."""
        source_id, e = self.episode_sources[e]
        reader = self.readers[source_id]
        episode = reader.episodes[e]
        cfg = self.window_config
        length = int(episode["length"])
        if not 0 <= k < length:
            raise IndexError("anchor is outside its episode")
        action_times = range(
            k, k + cfg.action_horizon * cfg.action_stride, cfg.action_stride
        )
        future_times = range(
            k + cfg.future_frame_stride,
            k + (cfg.future_frame_horizon + 1) * cfg.future_frame_stride,
            cfg.future_frame_stride,
        )
        quality = None
        if (
            self.dagger_quality_filter
            and "high_quality" in reader.columns[reader.data_paths[e]]
        ):
            # Action target action[t] uses the quality flag of its own row t.
            indices = sorted(
                {
                    k,
                    *(i for i in action_times if i < length),
                    *(i for i in future_times if i < length),
                }
            )
            table = reader.read_table(e, indices, ["high_quality"])
            flags = np.asarray(table["high_quality"].to_pylist()).reshape(-1)
            if len(flags) != len(table) or not np.isin(flags, [0, 1]).all():
                raise ValueError("high_quality must contain 0/1 or boolean values")
            quality = dict(zip(np.asarray(table["frame_index"]), flags))
            if quality[k] == 0:
                return None

        state_ids = history_indices(
            k, cfg.state_horizon, cfg.state_stride, cfg.history_pad_mode
        )
        action_ids = future_indices(
            action_times, length, cfg.action_pad_mode, quality
        )
        future_ids = future_indices(future_times, length, cfg.future_frame_pad_mode, quality)
        # States come from observation.state; action targets from the recorded
        # action column at the same target frames.
        wrist_state, hand_state = read_motion(reader, e, state_ids)
        wrist_action, hand_action = read_motion(reader, e, action_ids, column="action")
        sample = {
            "wrist_state": wrist_state,
            "hand_state": hand_state,
            "wrist_action": wrist_action,
            "hand_action": hand_action,
            "instruction": list(episode["instructions"]),
            "instruction_num": len(episode["instructions"]),
            "dataset_name": episode["tasks"][0],
            "episode_index": int(episode["episode_index"]),
            "__key__": f"episode_{episode['episode_index']}_frame_{k}",
        }
        cameras = ["head", "chest"] if self.load_chest else ["head"]
        camera_ids = [k] + future_ids if load_media else [k]
        for camera, (intrinsic, poses) in camera_parameters(
            reader, e, camera_ids, cameras
        ).items():
            prefix = "" if camera == "head" else "chest_"
            sample[f"{prefix}intrinsic"] = intrinsic
            sample[f"{prefix}extrinsic"] = poses[0]
            if load_media and future_ids:
                sample[f"future_{camera}_extrinsic"] = poses[1:]
        if load_media:
            self._read_window_media(sample, reader, e, k, future_ids, compressed)
        return sample

    # Gather history/future media together; resize and scale intrinsics once before JPEG encoding.
    def _read_window_media(self, sample, reader, e, k, future_ids, compressed=False):
        cfg = self.window_config
        image_ids = history_indices(
            k, cfg.image_horizon, cfg.image_stride, cfg.history_pad_mode
        )
        image_refs = [{} for _ in image_ids]
        future_refs = [{} for _ in future_ids]
        for camera in ["head", "chest"] if self.load_chest else ["head"]:
            prefix = "" if camera == "head" else "chest_"
            key = f"observation.images.{camera}"
            intrinsic = sample[f"{prefix}intrinsic"]
            if compressed and self.target_image_size is not None:
                height, width = reader.info["features"][key]["shape"][:2]
                target_h, target_w = self.target_image_size
                sx, sy = target_w / width, target_h / height
                intrinsic[0] *= sx
                intrinsic[1] *= sy
                intrinsic[2] *= sx
                intrinsic[3] *= sy
            sample[f"{prefix}intrinsic"] = intrinsic

            # Read history and future together, writing the final window format directly.
            images = reader.read_media(
                e,
                key,
                image_ids + future_ids,
                compressed=compressed,
                target_size=self.target_image_size,
            )
            if compressed:
                for ref, image in zip(image_refs + future_refs, images):
                    ref[f"{prefix}image.jpg"] = image
            else:
                sample[f"{prefix}image"] = images[: len(image_ids)]
                if future_ids:
                    sample[f"{prefix}future_frames"] = images[len(image_ids) :]
            if self.load_depth:
                depth = reader.read_media(
                    e,
                    f"{key}_depth",
                    image_ids,
                    compressed=compressed,
                    target_size=self.target_image_size,
                )
                if compressed:
                    for ref, frame in zip(image_refs, depth):
                        ref[f"{prefix}depth.npy"] = frame
                else:
                    sample[f"{prefix}depth"] = depth

        if compressed:
            sample["image_frame_refs"] = image_refs
            if future_refs:
                sample["future_frame_refs"] = future_refs

    # Choose active views after image augmentation, preserving the established RNG order.
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
        view_mask = np.array(["head" in active_views, "chest" in active_views], dtype=bool)
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

    # Check raw motion, apply geometry/normalization, then check the transformed model fields.
    def _process_motion(self, sample, extrinsic, normalizer=None):
        """Check raw motion before geometry, then validate transformed model inputs."""
        motion = {
            key: sample[key].astype(np.float32)
            for key in (
                "wrist_state",
                "hand_state",
                "wrist_action",
                "hand_action",
            )
        }
        self.checker.check(finite=motion)
        self.checker.check(
            rot6d={key: motion[key] for key in ("wrist_state", "wrist_action")},
            state_action_delta=tuple(motion.values()),
        )
        state, action = process_state_action(
            **motion,
            extrinsic=extrinsic,
            normalizer=normalizer,
            hand_ndim=self.hand_ndim,
            motion_type=self.motion_type,
            use_relative_action=self.use_relative_action,
        )
        self.checker.check(
            finite={"state": state, "action": action}, rot6d={"state": state, "action": action}
        )
        return state, action

    # Post-shuffle VLA stage: validate, transform, augment, choose instruction and pad supervision.
    def sample_to_data(self, sample):
        """Convert one LeRobot sample into the raw VLA sample schema used by the project.

        This is the dataset-side producer contract for `UnifiedVLACollator`.
        The returned mapping contains visual history, instruction text, intrinsic
        parameters, padded state/action tensors, action-valid masks, and bookkeeping
        fields such as `n_states`, `n_actions`, and `is_vla_data`.
        """
        self.checker.check(
            sample_schema=(
                sample,
                {
                    "required_keys": (
                        "wrist_state",
                        "hand_state",
                        "wrist_action",
                        "hand_action",
                        "extrinsic",
                        "intrinsic",
                        "instruction",
                        "instruction_num",
                        "image",
                    ),
                },
            )
        )

        # Validate calibration and motion before augmentation.
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

        state, action = self._process_motion(sample, extrinsic_raw, self.normalizer)

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
            raise ValueError("load_chest=True requires chest_image")
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

        state_pad = np.zeros((self.state_horizon, *state.shape[1:]), dtype=np.float32)
        state_pad[: state.shape[0]] = state
        action_pad = np.zeros((self.action_horizon, *action.shape[1:]), dtype=np.float32)
        actions_valid_mask = np.zeros((self.action_horizon, *action.shape[1:]), dtype=bool)
        action_pad[: action.shape[0]] = action
        actions_valid_mask[: action.shape[0]] = True

        data = self.build_raw_model_inputs(
            instruction=instruction,
            image=image,
            intrinsic=intrinsic,
            active_views=active_views,
            chest_image=chest_image,
            chest_intrinsic=chest_intrinsic,
        )

        data.update(
            {
                "states": state_pad,
                "n_states": np.array(state.shape[0], dtype=np.int32),
                "actions": action_pad,
                "actions_valid_mask": actions_valid_mask,
                "n_actions": np.array(action.shape[0], dtype=np.int32),
                "is_vla_data": np.array(True, dtype=bool),
            }
        )

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
                n_valid=n_valid,
                K=K,
            )
            data["future_head_motion"] = head_motion

            chest_ff, _ = pad_future(
                sample.get("chest_future_frames")
            )  # zero-filled when chest RGB is absent
            data["chest_future_frames"] = chest_ff
            # Gate on the extrinsics actually read below, not on chest RGB frames:
            # the two come from independent sources (load_chest vs meta["cameras"]).
            if sample.get("future_chest_extrinsic") is not None:
                chest_motion = compute_relative_motion_padded(
                    current_flat16=sample.get("chest_extrinsic"),
                    future_flat=sample.get("future_chest_extrinsic"),
                    n_valid=n_valid,
                    K=K,
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


class VLALowLevelLeRobotDataset(VLALeRobotDataset):
    """Scan all train-split anchors once, without video, thinning or padded rows."""

    lowdim_only = True

    def __init__(self, *args, **kwargs):
        kwargs.update(mode="val", val_stride=1)
        super().__init__(*args, **kwargs)

    # Normalizer scans use the same motion semantics without image decoding or padded target rows.
    def sample_to_data(self, sample):
        """Extract lowdim fields and compute state/action."""
        self.checker.check(
            sample_schema=(
                sample,
                {
                    "required_keys": (
                        "wrist_state",
                        "hand_state",
                        "wrist_action",
                        "hand_action",
                        "extrinsic",
                    ),
                },
            )
        )

        extrinsic = sample["extrinsic"].astype(np.float32).reshape(4, 4)
        self.checker.check(extrinsic=extrinsic)

        state, action = self._process_motion(sample, extrinsic)

        if not self.use_relative_action:
            return {
                "motions": np.concatenate([state, action], axis=0),
            }
        return {
            "states": state,
            "actions": action,
        }

    def get_collator(self):
        return ConcatDataCollator()


class UnifiedLeRobotDataset(UnifiedDataset):
    """LeRobot VLA stream under the shared unified batching contract."""
