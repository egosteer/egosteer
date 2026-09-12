"""LeRobot episode stream with EgoSteer's existing BC sample semantics."""

import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist

from src.utils.pytorch_util import dict_apply

from .lerobot_reader import LeRobotEpisodeReader
from .lerobot_schema import camera_parameters, unpack_motion
from .lerobot_stream import ResumableEpisodeStream
from .sanity_checks import DataSkipError, current_worker_id
from .unified_vla_collator import ConcatDataCollator
from .vla_dataset import VLAWdsDataset, VLALowLevelWdsDataset


def worker_partition():
    worker = torch.utils.data.get_worker_info()
    wid, nw = (worker.id, worker.num_workers) if worker else (0, 1)
    if dist.is_available() and dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world <= 0 or not 0 <= rank < world:
        raise ValueError(f"invalid rank/world size: {rank}/{world}")
    return rank * nw + wid, world * nw


def history_indices(k, horizon, stride, mode):
    ids = [k - i * stride for i in range(horizon - 1, -1, -1)]
    return [max(i, 0) for i in ids] if mode == "repeat" else [i for i in ids if i >= 0]


def future_indices(k, length, horizon, stride, mode, offset=0):
    ids = [k + offset + i * stride for i in range(horizon)]
    return [min(i, length - 1) for i in ids] if mode == "repeat" else [i for i in ids if i < length]


class VLALeRobotStreamDataset(VLAWdsDataset):
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
        return sum(max(0, int(ep["length"]) - 1) for ep in self.reader.episodes)

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

    def materialize(self, e, k):
        """Apply shared preprocessing and skip known data-quality failures."""
        sample = self.read_window(e, k, load_media=not self.lowdim_only)
        self.checker.note_sample_seen()
        start = time.perf_counter()
        try:
            data = self.sample_to_data(sample)
        except DataSkipError as exc:
            self.checker.log_skip(current_worker_id(), exc, sample)
            return None
        transform_s = time.perf_counter() - start
        data = dict_apply(data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
        if self.debug_profile_timing and not self.lowdim_only:
            data["debug_sample_profile"] = {
                "worker_id": current_worker_id(),
                "sample_to_data_s": transform_s,
                "preprocess_total_s": time.perf_counter() - start,
            }
        return data

    def materialize_with_context(self, e, k):
        """Attach the episode/frame locator to unexpected read or transform errors."""
        try:
            return self.materialize(e, k)
        except Exception as exc:
            episode_id = self.reader.episodes[e]["episode_index"]
            raise RuntimeError(
                f"LeRobot data error: root={self.root}, episode={episode_id}, frame={k}"
            ) from exc

    def iter_samples(self, global_worker, total_workers, logical_worker=None):
        if total_workers < 1 or not 0 <= global_worker < total_workers:
            raise ValueError("invalid worker partition")
        valid = np.array([e for e, ep in enumerate(self.reader.episodes) if ep["length"] > 1], dtype=np.int64)
        assigned = valid[global_worker::total_workers]
        if not len(assigned):
            if self.mode == "val":
                return
            raise ValueError("each training worker needs an episode; reduce workers or add data")
        if self.mode == "train":
            logical_worker = global_worker if logical_worker is None else logical_worker
            saved = self.worker_resume_states.get(logical_worker) if self.resume_enabled else None
            yield from ResumableEpisodeStream(self, assigned, global_worker, logical_worker, saved)
            return
        seen = 0
        for e in assigned:
            for k in range(int(self.reader.episodes[e]["length"]) - 1):
                keep = seen % self.val_stride == 0
                seen += 1
                if not keep:
                    continue
                sample = self.materialize_with_context(int(e), k)
                if sample is not None:
                    yield sample

    def __iter__(self):
        self.reader.reset_caches()
        worker = torch.utils.data.get_worker_info()
        wid, nw = (worker.id, worker.num_workers) if worker else (0, 1)
        logical = wid
        if self.mode == "train" and self.resume_enabled:
            if nw != self.resume_num_workers:
                raise ValueError("resume worker count differs from the configured DataLoader")
            # A new DataLoader begins at physical worker 0. Rotate logical
            # ownership to the worker that would deliver the next saved batch.
            logical = (wid + self.resume_batches) % nw
            global_worker = self.resume_rank * nw + logical
            total_workers = self.resume_world_size * nw
        else:
            global_worker, total_workers = worker_partition()
        yield from self.iter_samples(global_worker, total_workers, logical)

    def build_pipeline(self):
        return iter(self)

    def get_collator(self):
        collator = super().get_collator()
        if self.mode == "train" and self.resume_enabled:
            from .stream_checkpoint import StreamCheckpointCollator
            return StreamCheckpointCollator(collator)
        return collator

    def get_validation_dataset(self):
        dataset = deepcopy(self)
        dataset.mode = "val"
        dataset.aug_transform = False
        dataset.split = self.val_split
        dataset.resume_enabled = False
        dataset.worker_resume_states = {}
        dataset.reader = LeRobotEpisodeReader(self.root, self.val_split, **self.reader_kwargs)
        return dataset


class VLALowLevelLeRobotDataset(VLALeRobotStreamDataset):
    """Scan all train-split anchors once, without video, thinning or padded rows."""

    lowdim_only = True

    def __init__(self, *args, **kwargs):
        kwargs.update(mode="val", val_stride=1)
        super().__init__(*args, **kwargs)

    def sample_to_data(self, sample):
        return VLALowLevelWdsDataset.sample_to_data(self, sample)

    def get_collator(self):
        return ConcatDataCollator()
