"""LeRobot pipeline utilities for EgoSteer training.

Provides field mappings, Parquet/video readers, resumable episode streams
and shared VLA/VLM dataset lifecycle and checkpoint handling.
"""

import hashlib
import json
import math
import os
import pickle
import random
import time
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.utils.pytorch_util import dict_apply
from ..data_transforms import COLOR_AUG
from ..sanity_checks import DataSkipError, current_worker_id



MOTION_SLICES = {
    "left_arm_joints": slice(0, 7),
    "right_arm_joints": slice(7, 14),
    "left_hand_motors": slice(14, 20),
    "right_hand_motors": slice(20, 26),
    "left_wrist_pose": slice(26, 35),
    "right_wrist_pose": slice(35, 44),
    "left_fingertips": slice(44, 59),
    "right_fingertips": slice(59, 74),
}
CALIBRATION_SHAPES = {
    **{f"calibration.{cam}_intrinsics": (3, 3) for cam in ("head", "chest")},
    **{f"calibration.{cam}_world2cam": (4, 4) for cam in ("head", "chest")},
    **{f"calibration.{cam}_cam_to_{side}_base": (4, 4)
       for cam in ("head", "chest") for side in ("left", "right")},
}
FRAME_COLUMNS = ("observation.state", "action", "timestamp", "frame_index",
                 "episode_index", "index", "task_index")
VIDEO_KEYS = tuple(f"observation.images.{cam}{suffix}"
                   for cam in ("head", "chest") for suffix in ("", "_depth"))
EPISODE_COLUMNS = {
    "episode_index", "length", "tasks", "instructions", "dataset_from_index",
    "dataset_to_index", "data/chunk_index", "data/file_index", *CALIBRATION_SHAPES,
    *(f"videos/{key}/{field}" for key in VIDEO_KEYS
      for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")),
}


def camera_parameters(episode, camera):
    """Return float32 intrinsic4 and flattened world-to-camera extrinsic16."""
    intrinsic = episode[f"calibration.{camera}_intrinsics"]
    extrinsic = episode[f"calibration.{camera}_world2cam"]
    return (
        intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]].astype(np.float32),
        extrinsic.reshape(16).astype(np.float32),
    )


def unpack_motion(values):
    """Map 74D to wrist18/fingertips30; preserve the precomputed world-frame FK.

    Wrist order is [Lxyz, Rxyz, Lrot6d, Rrot6d], not the source's two 9D blocks.
    The 26 joint/motor channels are not used by the fingertip model.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 74:
        raise ValueError(f"expected [T,74] motion, got {values.shape}")
    left = values[:, MOTION_SLICES["left_wrist_pose"]]
    right = values[:, MOTION_SLICES["right_wrist_pose"]]
    wrist = np.concatenate([left[:, :3], right[:, :3], left[:, 3:], right[:, 3:]], axis=-1)
    hand = np.concatenate([
        values[:, MOTION_SLICES["left_fingertips"]],
        values[:, MOTION_SLICES["right_fingertips"]],
    ], axis=-1)
    return wrist, hand


def depth_parameters(feature):
    """Read persisted encoder parameters; never guess them from the codec."""
    info = {**(feature.get("video_info") or {}), **(feature.get("info") or {})}
    if not (info.get("is_depth_map") or info.get("video.is_depth_map")):
        raise ValueError("depth feature must declare is_depth_map=true")
    params = {}
    for key in ("depth_min", "depth_max", "shift", "use_log"):
        if f"video.{key}" not in info:
            raise ValueError(f"depth feature missing persisted video.{key}")
        params[key] = info[f"video.{key}"]
    lo, hi, shift = (float(params[k]) for k in ("depth_min", "depth_max", "shift"))
    if (not all(math.isfinite(v) for v in (lo, hi, shift)) or not 0 <= lo < hi
            or not isinstance(params["use_log"], bool)
            or (params["use_log"] and lo + shift <= 0)):
        raise ValueError(f"invalid depth quantization parameters: {params}")
    return params


def dequantize_depth_m(codes, *, depth_min, depth_max, shift, use_log):
    """Invert the official LeRobot 0.6.1 12-bit mapping, output float32 metres.

    Release specification §5.2 reserves zero as invalid. Restore these pixels
    to zero explicitly (upstream's generic inverse maps code zero to depth_min).
    Valid codes follow the same affine/logarithmic inverse as depth_utils.py.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.dtype != np.uint16 or np.any(codes > 4095):
        raise ValueError("expected uint16 HW depth codes in [0,4095]")
    lo, hi, shift = float(depth_min), float(depth_max), float(shift)
    if use_log:
        offset = math.log(lo + shift)
        scale = (math.log(hi + shift) - offset) / 4095
    else:
        offset, scale = lo, (hi - lo) / 4095
    result = codes.astype(np.float32) * scale + offset
    if use_log:
        np.exp(result, out=result)
        result -= shift
    np.clip(result, lo, hi, out=result)
    result[codes == 0] = 0.0
    return result


class VideoFrameCache:
    """Bounded frame LRU with a persistent sequential decoder per video file."""

    def __init__(self, max_frames=256, max_readers=4):
        if max_frames < 1 or max_readers < 1:
            raise ValueError("video cache sizes must be positive")
        self.max_frames = int(max_frames)
        self.max_readers = int(max_readers)
        self.frames = OrderedDict()
        self.readers = OrderedDict()

    def close(self):
        for state in self.readers.values():
            state["container"].close()
        self.readers.clear()
        self.frames.clear()

    def _get_reader(self, path, fps):
        if path not in self.readers:
            container = av.open(path)
            stream = container.streams.video[0]
            stream.thread_count = 1
            if stream.average_rate is None or not np.isclose(float(stream.average_rate), fps):
                container.close()
                raise ValueError(f"{path}: video fps does not match info.json fps={fps}")
            self.readers[path] = {
                "container": container,
                "stream": stream,
                "decoder": None,
                "last": -1,
            }
            while len(self.readers) > self.max_readers:
                self.readers.popitem(last=False)[1]["container"].close()
        self.readers.move_to_end(path)
        return self.readers[path]

    def _decode_interval(self, path, first, last, fps, depth):
        """Seek when needed, then decode forward while retaining decoder position."""
        state = self._get_reader(path, fps)
        stream = state["stream"]
        needs_seek = state["decoder"] is None or first <= state["last"] or first - state["last"] > fps
        if needs_seek:
            pts = int((first / fps) / float(stream.time_base))
            state["container"].seek(pts, stream=stream, backward=True, any_frame=False)
            state["decoder"] = state["container"].decode(stream)
            state["last"] = -1

        for frame in state["decoder"]:
            if frame.pts is None:
                raise ValueError(f"{path}: missing video PTS")
            position = float(frame.pts * stream.time_base) * fps
            index = round(position)
            if not np.isclose(position, index, atol=1e-3):
                raise ValueError(f"{path}: frame PTS is not aligned to the declared fps")
            state["last"] = index
            if index < first:
                continue
            if index > last:
                break
            if depth:
                if frame.format.name != "gray12le":
                    raise ValueError(f"{path}: expected gray12le depth, got {frame.format.name}")
                array = dequantize_depth_m(frame.to_ndarray(format="gray12le"), **depth)
            else:
                array = frame.to_ndarray(format="rgb24")
            yield index, array
            if index == last:
                break

    def read(self, path, indices, fps, depth=None):
        path = str(path)
        indices = [int(i) for i in indices]
        if not indices or min(indices) < 0:
            raise ValueError("video indices must be nonempty and nonnegative")
        kind = tuple(sorted(depth.items())) if depth else None
        need = sorted(set(indices))
        key = lambda i: (path, kind, i)
        resolved = {i: self.frames[key(i)] for i in need if key(i) in self.frames}
        for i in resolved:
            self.frames.move_to_end(key(i))
        missing = [i for i in need if i not in resolved]
        if missing:
            requested = set(missing)
            for i, array in self._decode_interval(path, missing[0], missing[-1], fps, depth):
                self.frames[key(i)] = array
                if i in requested:
                    resolved[i] = array
                while len(self.frames) > self.max_frames:
                    self.frames.popitem(last=False)
            if set(need) - resolved.keys():
                raise ValueError(f"{path}: missing video frames {sorted(set(need) - resolved.keys())}")
        return np.stack([resolved[i] for i in indices])


class LeRobotEpisodeReader:
    """Read one episode split with projected Parquet columns and worker-local caches."""

    frame_columns = FRAME_COLUMNS
    episode_columns = EPISODE_COLUMNS
    video_keys = VIDEO_KEYS

    def __init__(self, root, split, row_group_cache_size=8, frame_cache_size=256,
                 video_reader_cache_size=4):
        self.root = Path(root)
        self.split = split
        self.row_group_cache_size = int(row_group_cache_size)
        self.frame_cache_size = int(frame_cache_size)
        self.video_reader_cache_size = int(video_reader_cache_size)
        if min(self.row_group_cache_size, self.frame_cache_size, self.video_reader_cache_size) < 1:
            raise ValueError("reader cache sizes must be positive")
        self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        self._validate_features()
        self.depth_params = {
            key: depth_parameters(self.info["features"][key])
            for key in self.video_keys if key.endswith("_depth")
        }
        self.data_template = self.info["data_path"]
        self.video_template = self.info["video_path"] if self.video_keys else None
        self.tasks = self._read_tasks()
        self.episodes = self._read_episodes()
        self.blocks = self._index_row_groups()
        self.reset_caches()

    def _validate_features(self):
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("expected codebase_version v3.0")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("info.json fps must be positive")
        features = self.info["features"]
        for key in ("observation.state", "action"):
            if features[key]["shape"] != [74] or features[key]["dtype"] != "float32":
                raise ValueError(f"{key} must declare float32 [74], per release specification §4")
        for key in VIDEO_KEYS:
            channels = 1 if key.endswith("_depth") else 3
            shape = features[key]["shape"]
            if (features[key]["dtype"] != "video" or len(shape) != 3
                    or shape[-1] != channels or min(shape) < 1):
                raise ValueError(f"{key} must declare HWC video with {channels} channels")

    def _read_tasks(self):
        rows = pq.read_table(
            self.root / "meta/tasks.parquet", columns=["task_index", "task"],
        ).to_pylist()
        tasks = {int(row["task_index"]): row["task"] for row in rows}
        if len(tasks) != len(rows) or not all(isinstance(name, str) and name for name in tasks.values()):
            raise ValueError("invalid task vocabulary")
        return tasks

    def _read_episodes(self):
        """Project metadata before filtering the split; omit the large stats columns."""
        try:
            start, stop = map(int, self.info["splits"][self.split].split(":"))
        except (KeyError, ValueError, AttributeError) as exc:
            raise ValueError(
                f"info.json must declare split {self.split!r} as an episode interval start:stop"
            ) from exc
        if not 0 <= start <= stop <= int(self.info["total_episodes"]):
            raise ValueError(f"invalid episode split {self.split}: {start}:{stop}")
        metadata_paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not metadata_paths:
            raise ValueError("missing meta/episodes parquet files")
        episodes = []
        for path in metadata_paths:
            names = set(pq.read_schema(path).names)
            missing = self.episode_columns - names
            if missing:
                raise ValueError(f"{path}: missing release metadata {sorted(missing)}")
            columns = sorted(self.episode_columns | ({"split"} & names))
            table = pq.read_table(path, columns=columns)
            ids = np.asarray(table["episode_index"])
            table = table.take(pa.array(np.flatnonzero((ids >= start) & (ids < stop))))
            for row in table.to_pylist():
                self._validate_episode(row)
                episodes.append(row)
        episodes.sort(key=lambda row: int(row["episode_index"]))
        if [int(row["episode_index"]) for row in episodes] != list(range(start, stop)):
            raise ValueError(f"metadata has missing/duplicate episodes for split {self.split}")
        return episodes

    def _validate_episode(self, row):
        self._validate_episode_bounds(row)
        eid = row["episode_index"]
        if not isinstance(row["tasks"], list) or len(row["tasks"]) != 1 or row["tasks"][0] not in self.tasks.values():
            raise ValueError(f"episode {eid}: tasks must contain one vocabulary task name")
        if (not isinstance(row["instructions"], list) or not row["instructions"]
                or not all(isinstance(s, str) and s.strip() for s in row["instructions"])):
            raise ValueError(f"episode {eid}: instructions must be nonempty strings")
        # Keep all native instructions, including merged annotations.
        for field, shape in CALIBRATION_SHAPES.items():
            array = np.asarray(row[field], dtype=np.float64)
            if array.shape != (int(np.prod(shape)),) or not np.isfinite(array).all():
                raise ValueError(f"episode {eid}: {field} must be finite flattened {shape}")
            row[field] = array.reshape(shape)
        if not np.allclose(row["calibration.head_world2cam"], np.eye(4), atol=1e-6):
            raise ValueError(f"episode {eid}: head_world2cam must be identity")
        self._validate_video_intervals(row, self.video_keys)

    def _validate_episode_bounds(self, row):
        eid = row["episode_index"]
        if int(row["length"]) <= 0 or row["dataset_to_index"] - row["dataset_from_index"] != row["length"]:
            raise ValueError(f"episode {eid}: invalid length/global index interval")
        if "split" in row and row["split"] != self.split:
            raise ValueError(f"episode {eid}: split column disagrees with info.json")

    def _validate_video_intervals(self, row, keys):
        eid = row["episode_index"]
        for key in keys:
            prefix = f"videos/{key}"
            first, last = (float(row[f"{prefix}/{s}_timestamp"]) for s in ("from", "to"))
            if not (np.isfinite(first) and np.isfinite(last) and 0 <= first < last):
                raise ValueError(f"episode {eid}: invalid {key} time interval")
            if not np.isclose(first * self.fps, round(first * self.fps), atol=1e-3):
                raise ValueError(f"episode {eid}: {key} offset not aligned to fps")
            if not np.isclose((last - first) * self.fps, row["length"], atol=1e-3):
                raise ValueError(f"episode {eid}: {key} interval does not match episode length")

    def data_path(self, row):
        return self.root / self.data_template.format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])

    def _index_row_groups(self):
        """Use footer statistics only; no 74D columns materialized at startup."""
        result = {}
        files = {}
        for e, episode in enumerate(self.episodes):
            files.setdefault(str(self.data_path(episode)), []).append(e)
        for path, positions in files.items():
            with pq.ParquetFile(path) as file:
                missing = set(self.frame_columns) - set(file.schema_arrow.names)
                if missing:
                    raise ValueError(f"{path}: missing frame columns {sorted(missing)}")
                column = file.schema.names.index("episode_index")
                for e in positions:
                    eid = self.episodes[e]["episode_index"]
                    result[e] = []
                    for group in range(file.num_row_groups):
                        stats = file.metadata.row_group(group).column(column).statistics
                        if stats is None or not stats.has_min_max or stats.min <= eid <= stats.max:
                            result[e].append(group)
                    if not result[e]:
                        raise ValueError(f"{path}: episode {eid} has no candidate row groups")
        return result

    def reset_caches(self):
        for file in getattr(self, "files", {}).values():
            file.close()
        if hasattr(self, "video"):
            self.video.close()
        self.files = OrderedDict()
        self.row_groups = OrderedDict()
        self.video = VideoFrameCache(self.frame_cache_size, self.video_reader_cache_size)

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key not in ("files", "row_groups", "video")}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reset_caches()

    def _get_file(self, path):
        if path not in self.files:
            self.files[path] = pq.ParquetFile(path)
            while len(self.files) > self.row_group_cache_size:
                self.files.popitem(last=False)[1].close()
        self.files.move_to_end(path)
        return self.files[path]

    def _read_row_group(self, file, path, group, columns):
        key = (path, group, tuple(columns))
        if key not in self.row_groups:
            self.row_groups[key] = file.read_row_group(group, columns=columns)
            while len(self.row_groups) > self.row_group_cache_size:
                self.row_groups.popitem(last=False)
        self.row_groups.move_to_end(key)
        return self.row_groups[key]

    def read_lowdim(self, e, indices, column):
        """Return requested 74D rows in requested order, selecting in Arrow first."""
        if column not in ("observation.state", "action"):
            raise ValueError(f"not a motion column: {column}")
        rows = self.read_rows(e, indices, [column, "task_index"])
        episode = self.episodes[e]
        for k, row in rows.items():
            if self.tasks.get(int(row["task_index"])) != episode["tasks"][0]:
                raise ValueError(f"episode {episode['episode_index']} frame {k}: inconsistent task")
        array = np.asarray([rows[int(k)][column] for k in indices], dtype=np.float32)
        if array.shape != (len(indices), 74) or not np.isfinite(array).all():
            raise ValueError(f"{column}: expected finite [T,74] values")
        return array

    def read_rows(self, e, indices, columns):
        """Project requested frame fields, preserving episode/frame identity."""
        episode = self.episodes[e]
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= episode["length"]:
            raise IndexError("frame indices outside episode")
        path = str(self.data_path(episode))
        file = self._get_file(path)
        columns = sorted(set(columns) | {"episode_index", "frame_index", "index", "timestamp"})
        frame_column = file.schema.names.index("frame_index")
        tables = []
        for group in self.blocks[e]:
            stats = file.metadata.row_group(group).column(frame_column).statistics
            if stats is not None and stats.has_min_max and not np.any((indices >= stats.min) & (indices <= stats.max)):
                continue
            table = self._read_row_group(file, path, group, columns)
            select = (np.asarray(table["episode_index"]) == episode["episode_index"])
            select &= np.isin(np.asarray(table["frame_index"]), indices)
            if select.any():
                tables.append(table.take(pa.array(np.flatnonzero(select))))
        selected = [] if not tables else pa.concat_tables(tables).to_pylist()
        rows = {int(row["frame_index"]): row for row in selected}
        if len(rows) != len(selected) or set(indices) != set(rows):
            raise ValueError(f"episode {episode['episode_index']}: missing/duplicate frame indices")
        for k, row in rows.items():
            if (row["index"] != episode["dataset_from_index"] + k
                    or not np.isclose(row["timestamp"], k / self.fps, atol=1e-5)):
                raise ValueError(f"episode {episode['episode_index']} frame {k}: inconsistent index/time")
        return rows

    def read_media(self, e, key, indices):
        if key not in self.video_keys:
            raise ValueError(f"unsupported release video feature {key}")
        episode = self.episodes[e]
        if not len(indices) or min(indices) < 0 or max(indices) >= episode["length"]:
            raise IndexError("media frame indices outside episode")
        prefix = f"videos/{key}"
        path = self.root / self.video_template.format(
            video_key=key, chunk_index=episode[f"{prefix}/chunk_index"], file_index=episode[f"{prefix}/file_index"])
        first = round(float(episode[f"{prefix}/from_timestamp"]) * self.fps)
        # RGB/depth do NOT share file indices or offsets in the release.
        frames = self.video.read(path, [first + int(k) for k in indices], self.fps, self.depth_params.get(key))
        expected = self.info["features"][key]["shape"]
        if tuple(frames.shape[1:3]) != tuple(expected[:2]):
            raise ValueError(f"{key}: decoded resolution disagrees with info.json")
        return frames


class StreamSample(dict):
    """Keep checkpoint metadata outside the public model sample fields."""


@contextmanager
def sample_random_seed(seed):
    """Seed every stochastic CPU transform without changing caller RNG state.

    Albumentations 2 owns generators independent of numpy.random.seed. Older
    versions use the global Python/NumPy generators seeded here as well.
    """
    py_state, np_state = random.getstate(), np.random.get_state()
    aug_np = getattr(COLOR_AUG, "random_generator", None)
    aug_py = getattr(COLOR_AUG, "py_random", None)
    aug_seed = getattr(COLOR_AUG, "seed", None)
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if hasattr(COLOR_AUG, "set_random_seed"):
                COLOR_AUG.set_random_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if aug_np is not None:
            COLOR_AUG.set_random_state(aug_np, aug_py)
            COLOR_AUG.seed = aug_seed


class EpisodeSource:
    """Cursor of the next candidate, including dropped and invalid anchors."""

    def __init__(self, dataset, assigned, global_worker, saved=None):
        self.dataset = dataset
        self.assigned = assigned
        self.worker = global_worker
        self.round_idx = self.episode_pos = self.frame = 0
        self.attempted = self.usable = 0
        if saved is not None:
            for name in ("round_idx", "episode_pos", "frame", "attempted", "usable"):
                setattr(self, name, int(saved[name]))
        self._start_round()
        if saved is not None:
            self.drop_rng.bit_generator.state = deepcopy(saved["drop_rng"])

    def _start_round(self):
        seed = self.dataset.seed
        self.order = np.random.default_rng([seed, self.round_idx, self.worker, 2]).permutation(self.assigned)
        self.drop_rng = np.random.default_rng([seed, self.round_idx, self.worker, 0])

    def next_descriptor(self):
        while True:
            if self.episode_pos == len(self.order):
                if self.attempted and not self.usable:
                    raise ValueError("all attempted samples in this worker round failed sanity checks")
                self.round_idx += 1
                self.episode_pos = self.frame = self.attempted = self.usable = 0
                self._start_round()
            e = int(self.order[self.episode_pos])
            length = self.dataset.num_anchors(e)
            if self.frame == length:
                self.episode_pos += 1
                self.frame = 0
                continue
            k = self.frame
            self.frame += 1
            if self.drop_rng.random() < self.dataset.drop_ratio:
                continue
            self.attempted += 1
            seed = int(np.random.SeedSequence(
                [self.dataset.seed, self.round_idx, self.worker, e, k, 3]).generate_state(1)[0])
            return e, k, seed

    def state_dict(self):
        return {
            "round_idx": self.round_idx,
            "episode_pos": self.episode_pos,
            "frame": self.frame,
            "attempted": self.attempted,
            "usable": self.usable,
            "drop_rng": self.drop_rng.bit_generator.state,
        }


class ResumableEpisodeStream:
    """Shuffle complete samples while checkpointing lightweight descriptors.

    Restored residents decode lazily. New samples decode in source order before
    entering the queue. Queue and buffer positions always refer to the same sample.
    """

    def __init__(self, dataset, assigned, global_worker, logical_worker, saved=None):
        self.dataset = dataset
        self.source = EpisodeSource(dataset, assigned, global_worker, saved["source"] if saved else None)
        self.shuffle_rng = np.random.default_rng([dataset.seed, global_worker, 1])
        self.queue = list(saved["queue"]) if saved else []
        self.buffer = [None] * len(self.queue)
        self.delivered = int(saved["delivered"]) if saved else 0
        if saved:
            self.shuffle_rng.bit_generator.state = deepcopy(saved["shuffle_rng"])
        self.live_state = {"worker_id": logical_worker, "queue": self.queue}

    def materialize(self, descriptor):
        e, k, seed = descriptor
        with sample_random_seed(seed):
            return self.dataset.materialize_with_context(e, k)

    def append_source(self):
        while True:
            descriptor = self.source.next_descriptor()
            sample = self.materialize(descriptor)
            if sample is None:
                continue
            self.source.usable += 1
            self.queue.append(descriptor)
            self.buffer.append(sample)
            return

    def __iter__(self):
        while True:
            self.append_source()
            if len(self.buffer) < self.dataset.shuffle_buffer:
                self.append_source()
            if len(self.buffer) < self.dataset.shuffle_initial:
                continue
            pick = int(self.shuffle_rng.integers(len(self.buffer)))
            output = self.buffer[pick]
            if output is None:
                output = self.materialize(self.queue[pick])
                if output is None:
                    raise RuntimeError("checkpoint resident no longer passes validation; dataset/transforms changed")
            for buffer in (self.buffer, self.queue):
                buffer[pick] = buffer[-1]
                buffer.pop()
            self.delivered += 1
            if self.dataset.resume_enabled:
                self.live_state.update(
                    source=self.source.state_dict(),
                    shuffle_rng=self.shuffle_rng.bit_generator.state,
                    delivered=self.delivered,
                )
                output = StreamSample(output)
                # The collator freezes this shared state after the last sample.
                output.stream_state = self.live_state
                output.stream_name = self.dataset.stream_name
            yield output


STREAM_STATE_KEY = "_stream_state"


def _json_value(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_fingerprint(dataset, collator_config):
    """Identify metadata and preprocessing; data/video payloads must stay immutable."""
    description = dataset.resume_description() if hasattr(dataset, "resume_description") else {
        "info": dataset.reader.info,
        "episodes": dataset.reader.episodes,
        "tasks": dataset.reader.tasks,
        "shape_meta": dataset.shape_meta,
        "split": dataset.split,
        "use_relative_action": dataset.use_relative_action,
        "load_depth": dataset.load_depth,
        "load_chest": dataset.load_chest,
        "target_image_size": dataset.target_image_size,
        "depth_clip_range": dataset.depth_clip_range,
        "view_dropout": dataset.view_dropout,
        "sanity_checks": dataset.sanity_checks,
    }
    description["collator"] = collator_config
    digest = hashlib.sha256(json.dumps(_json_value(description), sort_keys=True).encode())
    if getattr(dataset, "normalizer", None) is not None:
        for name, tensor in sorted(dataset.normalizer.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class StreamCheckpoint:
    """Only record a state after its batch was passed through a training step.

    Prefetch states are never authoritative. Failed-gradient steps also consume
    data, so consumed_batches is separate from successful global/update steps.
    """

    def __init__(self, dataset, batch_size, num_workers, rank=0, world_size=1,
                 micro_batches_per_epoch=None, collator_config=None, gradient_accumulation_steps=1):
        if (batch_size < 1 or num_workers < 0 or not 0 <= rank < world_size
                or gradient_accumulation_steps < 1
                or (micro_batches_per_epoch is not None and micro_batches_per_epoch < 1)):
            raise ValueError("invalid stream loader topology")
        self.dataset = dataset
        self.spec = {
            "version": 1,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "rank": int(rank),
            "world_size": int(world_size),
            "seed": dataset.seed,
            "drop_ratio": dataset.drop_ratio,
            "shuffle_buffer": dataset.shuffle_buffer,
            "shuffle_initial": dataset.shuffle_initial,
            "micro_batches_per_epoch": micro_batches_per_epoch,
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "dataset_fingerprint": dataset_fingerprint(dataset, collator_config),
        }
        self.consumed_batches = 0
        self.workers = {}
        dataset.resume_enabled = True
        dataset.resume_num_workers = max(1, num_workers)
        dataset.resume_rank = int(rank)
        dataset.resume_world_size = int(world_size)
        dataset.resume_batches = 0
        dataset.worker_resume_states = {}

    @property
    def rank_key(self):
        return f"rank_{self.spec['rank']}"

    def record_consumed(self, encoded):
        if not encoded:
            raise ValueError("LeRobot training batch is missing its stream checkpoint state")
        state = pickle.loads(encoded)
        worker = self.validate_consumed(state)
        self.workers[worker] = state
        self.consumed_batches += 1

    def validate_consumed(self, state):
        nw = max(1, self.spec["num_workers"])
        worker = self.consumed_batches % nw
        if state["worker_id"] != worker:
            raise ValueError("unexpected worker batch order; exact resume requires in-order DataLoader")
        expected = ((self.consumed_batches // nw) + 1) * self.spec["batch_size"]
        if state["delivered"] != expected:
            raise ValueError("stream batch delivery count is inconsistent with consumed batches")
        return worker

    def state_dict(self):
        payload = {
            "spec": self.spec,
            "consumed_batches": self.consumed_batches,
            "workers": self.workers,
        }
        # Rank-local keys prevent DCP from deduplicating different worker queues.
        return {self.rank_key: pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)}

    def load_state_dict(self, state):
        payload = pickle.loads(state[self.rank_key])
        if payload["spec"] != self.spec:
            changed = [key for key in self.spec if payload["spec"].get(key) != self.spec[key]]
            raise ValueError(f"stream checkpoint mismatch: {', '.join(changed)}")
        consumed = int(payload["consumed_batches"])
        if consumed < 0:
            raise ValueError("negative consumed batch count")
        workers = payload["workers"]
        self._validate_workers(workers, consumed)
        self.consumed_batches = consumed
        self.workers = workers
        self.dataset.resume_batches = consumed
        self.dataset.worker_resume_states = workers

    def _validate_workers(self, workers, consumed):
        nw = max(1, self.spec["num_workers"])
        expected_workers = set(range(min(nw, consumed)))
        if set(workers) != expected_workers:
            raise ValueError("checkpoint worker set disagrees with consumed batches")
        for wid, item in workers.items():
            batches = (consumed + nw - 1 - wid) // nw
            if item["worker_id"] != wid or item["delivered"] != batches * self.spec["batch_size"]:
                raise ValueError("invalid worker delivery count in stream checkpoint")
            if len(item["queue"]) > self.spec["shuffle_buffer"] - 1:
                raise ValueError("checkpoint shuffle queue exceeds capacity")

    def require_checkpoint(self, checkpoint_path):
        from torch.distributed.checkpoint import FileSystemReader
        keys = FileSystemReader(checkpoint_path).read_metadata().state_dict_metadata
        if f"app.data_stream.{self.rank_key}" not in keys:
            raise ValueError(
                "checkpoint has no LeRobot stream state for this rank; exact data resume is unavailable. "
                "Use finetune_checkpoint_path for a weights-only restart from an older checkpoint.")


class StreamCheckpointCollator:
    """Freeze worker state before the next prefetch can mutate the live queue."""

    def __init__(self, collator, stream_names=None):
        self.collator = collator
        self.stream_names = stream_names

    def __call__(self, samples):
        states = [getattr(sample, "stream_state", None) for sample in samples]
        if not states or any(state is None for state in states):
            raise ValueError("resume collator requires pure resumable LeRobot samples")
        if len({state["worker_id"] for state in states}) != 1:
            raise ValueError("one batch must belong to one stream worker")
        if self.stream_names is not None:
            streams = {sample.stream_name: sample.stream_state for sample in samples}
            if set(streams) != set(self.stream_names):
                raise ValueError("mixed batch is missing a stream checkpoint")
            state = {"worker_id": states[-1]["worker_id"], "streams": streams}
        else:
            state = states[-1]
        result = self.collator(samples)
        result[STREAM_STATE_KEY] = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return result


class MixedStreamCheckpoint:
    """Resume fixed-ratio VLA/VLM batches at the joint consumption boundary."""

    def __init__(self, dataset, batch_size, **kwargs):
        vla_count = math.ceil(batch_size * dataset.vla_ratio)
        vlm_count = batch_size - vla_count
        if vla_count < 1 or vlm_count < 1:
            raise ValueError("mixed training requires at least one VLA and one VLM sample per batch")
        if dataset.batch_size != batch_size:
            raise ValueError("mixed dataset batch_size must match the DataLoader")
        if not hasattr(dataset.vlm_dataset, "resume_enabled"):
            raise ValueError("mixed resume requires a resumable VLMLeRobotDataset")
        self.batch_size = int(batch_size)
        self.vla_ratio = float(dataset.vla_ratio)
        self.streams = {
            "vla": StreamCheckpoint(dataset.vla_dataset, batch_size=vla_count, **kwargs),
            "vlm": StreamCheckpoint(dataset.vlm_dataset, batch_size=vlm_count, **kwargs),
        }

    @property
    def consumed_batches(self):
        counts = {stream.consumed_batches for stream in self.streams.values()}
        if len(counts) != 1:
            raise ValueError("VLA/VLM consumption boundaries disagree")
        return counts.pop()

    @property
    def rank_key(self):
        return self.streams["vla"].rank_key

    def record_consumed(self, encoded):
        if not encoded:
            raise ValueError("mixed batch has no stream checkpoint state")
        state = pickle.loads(encoded)
        if set(state.get("streams", {})) != set(self.streams):
            raise ValueError("mixed batch must contain VLA and VLM stream states")
        for name, stream in self.streams.items():
            if stream.validate_consumed(state["streams"][name]) != state["worker_id"]:
                raise ValueError("mixed batch contains different logical workers")
        for name, stream in self.streams.items():
            stream.record_consumed(pickle.dumps(state["streams"][name]))

    def state_dict(self):
        payload = {
            "format": "vla_vlm_v1", "batch_size": self.batch_size, "vla_ratio": self.vla_ratio,
            "streams": {name: stream.state_dict() for name, stream in self.streams.items()},
        }
        return {self.rank_key: pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)}

    def load_state_dict(self, state):
        payload = pickle.loads(state[self.rank_key])
        if (payload.get("format") != "vla_vlm_v1" or payload["batch_size"] != self.batch_size
                or payload["vla_ratio"] != self.vla_ratio):
            raise ValueError("mixed stream checkpoint format or VLA/VLM ratio mismatch")
        for name, stream in self.streams.items():
            stream.load_state_dict(payload["streams"][name])
        self.consumed_batches  # validate the shared boundary before starting workers

    def require_checkpoint(self, checkpoint_path):
        self.streams["vla"].require_checkpoint(checkpoint_path)


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


class LeRobotStreamMixin:
    """Common lifecycle; datasets define num_anchors, read_window and sample_to_data."""

    lowdim_only = False
    stream_name = "vla"

    def materialize(self, e, k):
        """Apply shared preprocessing and skip known data-quality failures."""
        self.checker.note_sample_seen()
        sample = {"dataset_name": self.root, "episode_index": self.reader.episodes[e]["episode_index"],
                  "__key__": f"frame_{k}"}
        try:
            sample = self.read_window(e, k, load_media=not self.lowdim_only)
            start = time.perf_counter()
            data = self.sample_to_data(sample)
        except DataSkipError as exc:
            self.checker.log_skip(current_worker_id(), exc, sample)
            return None
        transform_s = time.perf_counter() - start
        data = dict_apply(data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
        if getattr(self, "debug_profile_timing", False) and not self.lowdim_only:
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
        valid = np.array([e for e in range(len(self.reader.episodes)) if self.num_anchors(e) > 0], dtype=np.int64)
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
            for k in range(self.num_anchors(e)):
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
            return StreamCheckpointCollator(collator)
        return collator
