"""LeRobot episode stream with EgoSteer's existing BC sample semantics."""

from copy import deepcopy

import numpy as np

from .base_dataset import LeRobotStreamMixin
from .lerobot_dataset import LeRobotEpisodeReader
from .schema import camera_parameters, unpack_motion
from ..unified_vla_collator import ConcatDataCollator
from ..unified_dataset import UnifiedDataset
from ..wds.vla_dataset import VLAWdsDataset, VLALowLevelWdsDataset


def history_indices(k, horizon, stride, mode):
    ids = [k - i * stride for i in range(horizon - 1, -1, -1)]
    return [max(i, 0) for i in ids] if mode == "repeat" else [i for i in ids if i >= 0]


def future_indices(k, length, horizon, stride, mode, offset=0):
    ids = [k + offset + i * stride for i in range(horizon)]
    return [min(i, length - 1) for i in ids] if mode == "repeat" else [i for i in ids if i < length]


class VLALeRobotDataset(LeRobotStreamMixin, VLAWdsDataset):
    """Sequential LeRobot windows with EgoSteer's shared transforms and collator.

    Workers own disjoint episodes and shuffle complete samples. Action targets
    are next measured states; the final frame has no successor. Train and val
    use the episode splits in the same root's info.json.
    """

    lowdim_only = False

    def __init__(self, root, shape_meta, split="train", val_split="val", seed=0,
                 drop_ratio=0.8, shuffle_buffer=4096, shuffle_initial=4096,
                 reader_kwargs=None, **kwargs):
        if not 0 <= drop_ratio < 1 or seed < 0:
            raise ValueError("drop_ratio must be in [0,1) and seed nonnegative")
        if shuffle_buffer < 1 or (shuffle_initial is not None and shuffle_initial < 1):
            raise ValueError("shuffle capacities must be positive")
        for key in ("wds_datasets", "val_wds_datasets", "keep_ratio", "dagger_quality_filter"):
            if key in kwargs:
                raise TypeError(f"{key} is a WDS option, not part of the LeRobot release stream")
        super().__init__(wds_datasets=[], shape_meta=shape_meta, shuffle_buffer=shuffle_buffer,
                         shuffle_initial=shuffle_initial, dagger_quality_filter=False, **kwargs)
        self._validate_config()
        self.root = str(root)
        self.split = split
        self.val_split = val_split
        self.seed = int(seed)
        self.drop_ratio = float(drop_ratio)
        self.shuffle_initial = min(shuffle_buffer, shuffle_initial or shuffle_buffer)
        self.reader_kwargs = dict(reader_kwargs or {})
        self.resume_enabled = False
        self.resume_batches = 0
        self.resume_num_workers = 1
        self.resume_rank = 0
        self.resume_world_size = 1
        self.worker_resume_states = {}
        self.reader = LeRobotEpisodeReader(root, split, **self.reader_kwargs)
        if not np.isclose(self.reader.fps, self.video_base_fps):
            raise ValueError("video_base_fps must match info.json fps")
        if self.mode == "train" and not len(self):
            raise ValueError("training split has no anchors with a next state")

    def _validate_config(self):
        if self.mode not in ("train", "val"):
            raise ValueError("mode must be train or val")
        if self.motion_type != "fingertips" or self.action_ndim != 48 or self.hand_ndim != 15:
            raise ValueError("release adapter feeds the existing 48D fingertip model")
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
        return sum(self.num_anchors(e) for e in range(len(self.reader.episodes)))

    def num_anchors(self, e):
        return max(0, int(self.reader.episodes[e]["length"]) - 1)

    def read_window(self, e, k, load_media=True):
        """Build the raw wrist/hand window consumed by sample_to_data."""
        episode = self.reader.episodes[e]
        cfg = self.window_config
        length = int(episode["length"])
        if not 0 <= k < length - 1:
            raise IndexError("anchor must have a next state within its episode")
        state_ids = history_indices(k, cfg.state_horizon, cfg.state_stride, cfg.history_pad_mode)
        action_ids = future_indices(k, length, cfg.action_horizon, cfg.action_stride,
                                    cfg.action_pad_mode, offset=1)
        wrist_state, hand_state = unpack_motion(self.reader.read_lowdim(e, state_ids, "observation.state"))
        # Training targets come from next state, not the native action column.
        wrist_action, hand_action = unpack_motion(self.reader.read_lowdim(e, action_ids, "observation.state"))
        intrinsic, extrinsic = camera_parameters(episode, "head")
        sample = {
            "wrist_state": wrist_state,
            "hand_state": hand_state,
            "wrist_action": wrist_action,
            "hand_action": hand_action,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "instruction": list(episode["instructions"]),
            "instruction_num": len(episode["instructions"]),
            "dataset_name": episode["tasks"][0],
            "episode_index": int(episode["episode_index"]),
            "__key__": f"episode_{episode['episode_index']}_frame_{k}",
        }
        if load_media:
            self._read_window_media(sample, e, k)
        return sample

    def _read_window_media(self, sample, e, k):
        episode = self.reader.episodes[e]
        cfg = self.window_config
        image_ids = history_indices(k, cfg.image_horizon, cfg.image_stride, cfg.history_pad_mode)
        future_ids = future_indices(
            k, int(episode["length"]), cfg.future_frame_horizon, cfg.future_frame_stride,
            cfg.future_frame_pad_mode, offset=cfg.future_frame_stride,
        )
        cameras = ["head", "chest"] if self.load_chest else ["head"]
        for camera in cameras:
            prefix = "" if camera == "head" else "chest_"
            key = f"observation.images.{camera}"
            intrinsic, extrinsic = camera_parameters(episode, camera)
            sample[f"{prefix}image"] = self.reader.read_media(e, key, image_ids)
            if self.load_depth:
                # Reader depth is already float32 metres.
                sample[f"{prefix}depth"] = self.reader.read_media(e, f"{key}_depth", image_ids)
            if camera == "chest":
                sample["chest_intrinsic"] = intrinsic
                sample["chest_extrinsic"] = extrinsic
            if future_ids:
                sample[f"{prefix}future_frames"] = self.reader.read_media(e, key, future_ids)
                sample[f"future_{camera}_extrinsic"] = np.repeat(
                    extrinsic[None], len(future_ids), axis=0,
                )


    def get_validation_dataset(self):
        dataset = deepcopy(self)
        dataset.mode = "val"
        dataset.aug_transform = False
        dataset.split = self.val_split
        dataset.resume_enabled = False
        dataset.worker_resume_states = {}
        dataset.reader = LeRobotEpisodeReader(self.root, self.val_split, **self.reader_kwargs)
        return dataset


class VLALowLevelLeRobotDataset(VLALeRobotDataset):
    """Scan all train-split anchors once, without video, thinning or padded rows."""

    lowdim_only = True

    def __init__(self, *args, **kwargs):
        kwargs.update(mode="val", val_stride=1)
        super().__init__(*args, **kwargs)

    def sample_to_data(self, sample):
        return VLALowLevelWdsDataset.sample_to_data(self, sample)

    def get_collator(self):
        return ConcatDataCollator()


class UnifiedLeRobotDataset(UnifiedDataset):
    """Unified LeRobot VLA/VLM stream with joint checkpoint boundaries."""

    def get_collator(self):
        collator = super().get_collator()
        if self.mode == "train" and self.vlm_dataset is not None and self.vla_dataset.resume_enabled:
            from .checkpoint import StreamCheckpointCollator
            if isinstance(collator, StreamCheckpointCollator):
                collator = collator.collator
            return StreamCheckpointCollator(collator, stream_names=("vla", "vlm"))
        return collator
