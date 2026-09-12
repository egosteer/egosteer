"""LeRobot pipeline utilities for EgoSteer training.

Provides field mappings, Parquet/video readers, resumable episode streams
and shared VLA/VLM dataset lifecycle and checkpoint handling.
"""

import hashlib
import io
import json
import math
import os
import pickle
import random
import time
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.utils.pytorch_util import dict_apply
from ..data_transforms import COLOR_AUG, resize_frames
from ..sanity_checks import DataChecker, DataSkipError, current_worker_id
from ..unified_vla_collator import UnifiedVLACollator


CALIBRATION_SHAPES = {
    **{f"calibration.{cam}_intrinsics": (3, 3) for cam in ("head", "chest")},
    **{f"calibration.{cam}_world2cam": (4, 4) for cam in ("head", "chest")},
    **{
        f"calibration.{cam}_cam_to_{side}_base": (4, 4)
        for cam in ("head", "chest")
        for side in ("left", "right")
    },
}
VIDEO_KEYS = tuple(
    f"observation.images.{cam}{suffix}" for cam in ("head", "chest") for suffix in ("", "_depth")
)
EPISODE_COLUMNS = {
    "episode_index",
    "length",
    "tasks",
    "instructions",
    "dataset_from_index",
    "dataset_to_index",
    "data/chunk_index",
    "data/file_index",
    *CALIBRATION_SHAPES,
    *(
        f"videos/{key}/{field}"
        for key in VIDEO_KEYS
        for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ),
}


def camera_parameters(episode, camera):
    """Return float32 intrinsic4 and flattened world-to-camera extrinsic16."""
    intrinsic = episode[f"calibration.{camera}_intrinsics"]
    extrinsic = episode[f"calibration.{camera}_world2cam"]
    return (
        intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]].astype(np.float32),
        extrinsic.reshape(16).astype(np.float32),
    )


def read_motion(reader, e, indices):
    """Read measured 74D states and map them to wrist18/fingertips30."""
    table = reader.read_table(e, indices, ["observation.state", "task_index"])
    episode = reader.episodes[e]
    tasks = (reader.tasks.get(int(index)) for index in np.unique(np.asarray(table["task_index"])))
    if any(task != episode["tasks"][0] for task in tasks):
        raise ValueError(f"episode {episode['episode_index']}: inconsistent task")
    frame_ids = np.asarray(table["frame_index"])
    order = np.argsort(frame_ids)
    positions = order[np.searchsorted(frame_ids[order], indices)]
    # Arrow yields NumPy row views; avoid boxing every float through to_pylist().
    values = np.stack(table["observation.state"].to_numpy())[positions].astype(
        np.float32, copy=False
    )
    left, right = values[:, 26:35], values[:, 35:44]
    wrist = np.concatenate([left[:, :3], right[:, :3], left[:, 3:], right[:, 3:]], axis=-1)
    return wrist, values[:, 44:74].copy()


def depth_parameters(feature):
    """Read persisted encoder parameters; never guess them from the codec."""
    info = {**(feature.get("video_info") or {}), **(feature.get("info") or {})}
    params = {key: info[f"video.{key}"] for key in ("depth_min", "depth_max", "shift", "use_log")}
    lo, hi, shift = (float(params[k]) for k in ("depth_min", "depth_max", "shift"))
    if (
        not all(math.isfinite(v) for v in (lo, hi, shift))
        or not 0 <= lo < hi
        or not isinstance(params["use_log"], bool)
        or (params["use_log"] and lo + shift <= 0)
    ):
        raise ValueError(f"invalid depth quantization parameters: {params}")
    return params


def dequantize_depth_m(codes, *, depth_min, depth_max, shift, use_log):
    """Invert the official LeRobot 0.6.1 12-bit mapping, output float32 metres.

    Release specification §5.2 reserves zero as invalid. Restore these pixels
    to zero explicitly (upstream's generic inverse maps code zero to depth_min).
    Valid codes follow the same affine/logarithmic inverse as depth_utils.py.
    """
    codes = np.asarray(codes)
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


@dataclass
class WindowConfig:
    """Sampling window parameters for sliding window compose."""

    action_horizon: int = 32
    action_stride: int = 1
    state_horizon: int = 16
    state_stride: int = 2
    image_horizon: int = 1
    image_stride: int = 30
    history_pad_mode: str = "repeat"
    action_pad_mode: str = "truncate"
    future_frame_horizon: int = 0
    future_frame_stride: int = 30
    future_frame_pad_mode: str = "repeat"

    def __post_init__(self):
        valid_modes = {"repeat", "truncate"}
        for name in ("history_pad_mode", "action_pad_mode", "future_frame_pad_mode"):
            value = getattr(self, name)
            if value not in valid_modes:
                raise ValueError(f"Invalid {name}: {value}")


def decode_sample_media(sample):
    """Decode VLA frame windows or VLM image fields after shuffle."""
    if "image_frame_refs" not in sample:
        from PIL import Image

        for key, value in list(sample.items()):
            if key.startswith("image_") and key.endswith(".jpg"):
                sample[key] = Image.open(io.BytesIO(value)).convert("RGB")
        return sample

    for refs_key, fields in (
        (
            "image_frame_refs",
            (
                ("image", "image.jpg"),
                ("depth", "depth.npy"),
                ("chest_image", "chest_image.jpg"),
                ("chest_depth", "chest_depth.npy"),
            ),
        ),
        (
            "future_frame_refs",
            (("future_frames", "image.jpg"), ("chest_future_frames", "chest_image.jpg")),
        ),
    ):
        refs = sample.pop(refs_key, None)
        if refs is None:
            continue
        for output_key, media_key in fields:
            if refs[-1].get(media_key) is None:
                continue
            frames = []
            for ref in refs:
                value = ref[media_key]
                if media_key.endswith(".npy"):
                    frame = np.load(io.BytesIO(value))
                else:
                    frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("JPEG decoding failed")
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=frame)
                frames.append(frame)
            sample[output_key] = np.stack(frames)
    return sample


JPEG_QUALITY = 80


def encode_frame(array, target_size=None):
    """Resize and encode RGB as JPEG80 or depth as lossless NPY bytes."""
    if target_size is not None and tuple(array.shape[:2]) != tuple(target_size):
        interpolation = cv2.INTER_NEAREST if array.ndim == 2 else cv2.INTER_LINEAR
        array = resize_frames(array[None], target_size, interpolation=interpolation)[0]
    if array.ndim == 2:
        output = io.BytesIO()
        np.save(output, array, allow_pickle=False)
        return output.getvalue()
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


class LeRobotEpisodeReader:
    """Read one episode split with projected Parquet columns and worker-local caches."""

    is_vla = True
    episode_columns = EPISODE_COLUMNS
    video_keys = VIDEO_KEYS

    def __init__(
        self, root, split, row_group_cache_size=8, frame_cache_size=256, video_reader_cache_size=4
    ):
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
            for key in self.video_keys
            if key.endswith("_depth")
        }
        self.data_template = self.info["data_path"]
        self.video_template = self.info["video_path"] if self.video_keys else None
        self.tasks = {}
        if self.is_vla:
            rows = pq.read_table(
                self.root / "meta/tasks.parquet", columns=["task_index", "task"]
            ).to_pylist()
            self.tasks = {int(row["task_index"]): row["task"] for row in rows}
        self.episodes = self._read_episodes()
        self.data_paths = [
            str(
                self.root
                / self.data_template.format(
                    chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
                )
            )
            for row in self.episodes
        ]
        self.blocks, self.columns = {}, {}
        files = {}
        for e, path in enumerate(self.data_paths):
            files.setdefault(path, []).append(e)
        for path, positions in files.items():
            with pq.ParquetFile(path) as file:
                self.columns[path] = set(file.schema_arrow.names)
                column = file.schema.names.index("episode_index")
                stats = [
                    file.metadata.row_group(g).column(column).statistics
                    for g in range(file.num_row_groups)
                ]
                episode_ids = np.array([self.episodes[e]["episode_index"] for e in positions])
                self.blocks.update((e, []) for e in positions)
                for group, stat in enumerate(stats):
                    lo, hi = (
                        (0, len(positions))
                        if stat is None or not stat.has_min_max
                        else (
                            np.searchsorted(episode_ids, stat.min, side="left"),
                            np.searchsorted(episode_ids, stat.max, side="right"),
                        )
                    )
                    for e in positions[lo:hi]:
                        self.blocks[e].append(group)
        self.reset_caches()

    def _validate_features(self):
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("info.json fps must be positive")
        # Layout is fixed; sample values are checked by DataChecker after shuffle.
        if self.info["features"]["observation.state"]["shape"] != [74]:
            raise ValueError("observation.state must declare the released 74D layout")

    def _read_episodes(self):
        """Project metadata before filtering the split; omit the large stats columns."""
        start, stop = map(int, self.info["splits"][self.split].split(":"))
        if not 0 <= start <= stop <= int(self.info["total_episodes"]):
            raise ValueError(f"invalid episode split {self.split}: {start}:{stop}")
        metadata_paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        episodes = []
        for path in metadata_paths:
            names = set(pq.read_schema(path).names)
            columns = sorted(self.episode_columns | ({"split"} & names))
            table = pq.read_table(path, columns=columns)
            ids = np.asarray(table["episode_index"])
            table = table.take(pa.array(np.flatnonzero((ids >= start) & (ids < stop))))
            for row in table.to_pylist():
                eid = row["episode_index"]
                if (
                    int(row["length"]) <= 0
                    or row["dataset_to_index"] - row["dataset_from_index"] != row["length"]
                ):
                    raise ValueError(f"episode {eid}: invalid length/global index interval")
                if self.is_vla:
                    for field, shape in CALIBRATION_SHAPES.items():
                        row[field] = np.asarray(row[field], dtype=np.float64).reshape(shape)
                    if not np.allclose(row["calibration.head_world2cam"], np.eye(4), atol=1e-6):
                        raise ValueError(f"episode {eid}: head_world2cam must be identity")
                for key in self.video_keys:
                    prefix = f"videos/{key}"
                    if not self.is_vla and row.get(f"{prefix}/file_index") is None:
                        continue
                    first, last = (
                        float(row[f"{prefix}/{side}_timestamp"]) for side in ("from", "to")
                    )
                    if (
                        not (np.isfinite(first) and np.isfinite(last) and 0 <= first < last)
                        or not np.isclose(first * self.fps, round(first * self.fps), atol=1e-3)
                        or not np.isclose((last - first) * self.fps, row["length"], atol=1e-3)
                    ):
                        raise ValueError(
                            f"episode {eid}: {key} interval does not align with episode frames"
                        )
                episodes.append(row)
        episodes.sort(key=lambda row: int(row["episode_index"]))
        if [int(row["episode_index"]) for row in episodes] != list(range(start, stop)):
            raise ValueError(f"metadata has missing/duplicate episodes for split {self.split}")
        return episodes

    def reset_caches(self):
        for file in getattr(self, "files", {}).values():
            file.close()
        for reader in getattr(self, "video_readers", {}).values():
            reader["container"].close()
        self.files = OrderedDict()
        self.row_groups = OrderedDict()
        self.video_readers = OrderedDict()

    def __getstate__(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("files", "row_groups", "video_readers")
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reset_caches()

    def read_table(self, e, indices, columns):
        """Project requested frame fields, preserving episode/frame identity."""
        episode = self.episodes[e]
        indices = np.asarray(indices, dtype=np.int64)
        path = self.data_paths[e]
        if path not in self.files:
            self.files[path] = pq.ParquetFile(path)
            while len(self.files) > self.row_group_cache_size:
                self.files.popitem(last=False)[1].close()
        self.files.move_to_end(path)
        file = self.files[path]
        columns = sorted(set(columns) | {"episode_index", "frame_index", "index", "timestamp"})
        frame_column = file.schema.names.index("frame_index")
        tables = []
        requested = np.unique(indices)
        for group in self.blocks[e]:
            stats = file.metadata.row_group(group).column(frame_column).statistics
            if (
                stats is not None
                and stats.has_min_max
                and not np.any((indices >= stats.min) & (indices <= stats.max))
            ):
                continue
            cache_key = (path, group, tuple(columns))
            if cache_key not in self.row_groups:
                table = file.read_row_group(group, columns=columns)
                episode_ids = np.asarray(table["episode_index"])
                frame_ids = np.asarray(table["frame_index"])
                order = np.lexsort((frame_ids, episode_ids))
                ids = np.empty(
                    len(order), dtype=[("episode", episode_ids.dtype), ("frame", frame_ids.dtype)]
                )
                ids["episode"], ids["frame"] = episode_ids[order], frame_ids[order]
                self.row_groups[cache_key] = (table, ids, order)
                while len(self.row_groups) > self.row_group_cache_size:
                    self.row_groups.popitem(last=False)
            self.row_groups.move_to_end(cache_key)
            table, ids, order = self.row_groups[cache_key]
            # Match by episode/frame, then validate global index; malformed duplicates must stay visible.
            query = np.empty(len(requested), dtype=ids.dtype)
            query["episode"], query["frame"] = episode["episode_index"], requested
            left = np.searchsorted(ids, query, side="left")
            right = np.searchsorted(ids, query, side="right")
            positions = [order[lo:hi] for lo, hi in zip(left, right) if lo < hi]
            if positions:
                tables.append(table.take(pa.array(np.concatenate(positions))))
        if not tables:
            raise ValueError(f"episode {episode['episode_index']}: missing frame indices")
        selected = pa.concat_tables(tables)
        frame_ids = np.asarray(selected["frame_index"])
        unique_ids = np.unique(frame_ids)
        if len(unique_ids) != len(frame_ids) or not np.array_equal(unique_ids, np.unique(indices)):
            raise ValueError(f"episode {episode['episode_index']}: missing/duplicate frame indices")
        valid = np.asarray(selected["episode_index"]) == episode["episode_index"]
        valid &= np.asarray(selected["index"]) == episode["dataset_from_index"] + frame_ids
        valid &= np.isclose(np.asarray(selected["timestamp"]), frame_ids / self.fps, atol=1e-5)
        if not valid.all():
            k = frame_ids[np.flatnonzero(~valid)[0]]
            raise ValueError(
                f"episode {episode['episode_index']} frame {k}: inconsistent index/time"
            )
        return selected

    def read_media(self, e, key, indices, compressed=False, target_size=None):
        episode = self.episodes[e]
        prefix = f"videos/{key}"
        path = self.root / self.video_template.format(
            video_key=key,
            chunk_index=episode[f"{prefix}/chunk_index"],
            file_index=episode[f"{prefix}/file_index"],
        )
        first = round(float(episode[f"{prefix}/from_timestamp"]) * self.fps)
        # RGB/depth do NOT share file indices or offsets in the release.
        indices = [first + int(k) for k in indices]
        path = str(path)
        depth = self.depth_params.get(key)
        kind = tuple(sorted(depth.items())) if depth else None
        size = tuple(target_size) if compressed and target_size is not None else None
        cache_key = (path, kind, size)
        if cache_key not in self.video_readers:
            container = av.open(path)
            stream = container.streams.video[0]
            stream.thread_count = 1
            if stream.average_rate is None or not np.isclose(float(stream.average_rate), self.fps):
                container.close()
                raise ValueError(f"{path}: video fps does not match info.json")
            self.video_readers[cache_key] = dict(
                container=container, stream=stream, decoder=None, last=-1, frames=OrderedDict()
            )
            while len(self.video_readers) > self.video_reader_cache_size:
                self.video_readers.popitem(last=False)[1]["container"].close()
        self.video_readers.move_to_end(cache_key)
        reader = self.video_readers[cache_key]
        cache = reader["frames"]
        needed = sorted(set(indices))
        resolved = {i: cache[i] for i in needed if i in cache}
        for i in resolved:
            cache.move_to_end(i)
        missing = [i for i in needed if i not in resolved]
        if missing:
            stream = reader["stream"]
            first, last = missing[0], missing[-1]
            if (
                reader["decoder"] is None
                or first <= reader["last"]
                or first - reader["last"] > self.fps
            ):
                pts = int((first / self.fps) / float(stream.time_base))
                reader["container"].seek(pts, stream=stream, backward=True, any_frame=False)
                reader["decoder"] = reader["container"].decode(stream)
                reader["last"] = -1
            else:
                # Retain intermediate frames too: later history windows will request them.
                first = reader["last"] + 1
            requested = set(missing)
            for frame in reader["decoder"]:
                position = float(frame.pts * stream.time_base) * self.fps
                i = round(position)
                if not np.isclose(position, i, atol=1e-3):
                    raise ValueError(f"{path}: frame PTS is not aligned to fps")
                reader["last"] = i
                if i < first:
                    continue
                if i > last:
                    break
                if depth:
                    if frame.format.name != "gray12le":
                        raise ValueError(
                            f"{path}: expected gray12le depth, got {frame.format.name}"
                        )
                    array = dequantize_depth_m(frame.to_ndarray(format="gray12le"), **depth)
                else:
                    array = frame.to_ndarray(format="rgb24")
                source_shape = array.shape[:2]
                if size is not None:
                    interpolation = cv2.INTER_NEAREST if depth else cv2.INTER_LINEAR
                    array = resize_frames(array[None], size, interpolation=interpolation)[0]
                cache[i] = (array, source_shape)
                cache.move_to_end(i)
                if i in requested:
                    resolved[i] = cache[i]
                while len(cache) > self.frame_cache_size:
                    cache.popitem(last=False)
                if i == last:
                    break
        expected_shape = tuple(self.info["features"][key]["shape"][:2])
        if any(tuple(resolved[i][1]) != expected_shape for i in needed):
            raise ValueError(f"{key}: decoded resolution disagrees with info.json")
        if compressed:
            # JPEG/NPY bytes are local to this window, never shared across samples.
            encoded = {i: encode_frame(resolved[i][0]) for i in needed}
            return [encoded[i] for i in indices]
        return np.stack([resolved[i][0] for i in indices])


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


def build_lerobot_pipeline(dataset, global_worker, total_workers, logical_worker=None):
    """Read/encode -> shuffle -> decode/preprocess, with resumable source and queue."""
    episodes = [e for e in range(len(dataset.reader.episodes)) if dataset.num_anchors(e) > 0]
    assigned = np.asarray(episodes[global_worker::total_workers], dtype=np.int64)
    if not len(assigned):
        if dataset.mode == "val":
            return
        raise ValueError("each training worker needs an episode; reduce workers or add data")

    def read_window(descriptor):
        e, k, _ = descriptor
        try:
            return dataset.read_window(e, k, load_media=not dataset.lowdim_only, compressed=True)
        except DataSkipError as exc:
            dataset.checker.note_sample_seen()
            sample = {
                "dataset_name": dataset.root,
                "episode_index": dataset.reader.episodes[e]["episode_index"],
                "__key__": f"frame_{k}",
            }
            dataset.checker.log_skip(current_worker_id(), exc, sample)
        except Exception as exc:
            episode_id = dataset.reader.episodes[e]["episode_index"]
            raise RuntimeError(
                f"LeRobot read error: root={dataset.root}, episode={episode_id}, frame={k}"
            ) from exc

    def preprocess(descriptor, sample):
        e, k, seed = descriptor
        dataset.checker.note_sample_seen()
        sample = dict(sample)
        try:
            with sample_random_seed(seed) if seed is not None else nullcontext():
                if not dataset.lowdim_only:
                    sample = decode_sample_media(sample)
                start = time.perf_counter()
                data = dataset.sample_to_data(sample)
                transform_s = time.perf_counter() - start
                data = dict_apply(
                    data, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
                )
                if getattr(dataset, "debug_profile_timing", False) and not dataset.lowdim_only:
                    data["debug_sample_profile"] = {
                        "worker_id": current_worker_id(),
                        "sample_to_data_s": transform_s,
                        "preprocess_total_s": time.perf_counter() - start,
                    }
                return data
        except DataSkipError as exc:
            dataset.checker.log_skip(current_worker_id(), exc, sample)
        except Exception as exc:
            episode_id = dataset.reader.episodes[e]["episode_index"]
            raise RuntimeError(
                f"LeRobot sample error: root={dataset.root}, episode={episode_id}, frame={k}"
            ) from exc

    if dataset.mode == "val":
        seen = 0
        for e in assigned:
            for k in range(dataset.num_anchors(e)):
                keep = seen % dataset.val_stride == 0
                seen += 1
                if not keep:
                    continue
                descriptor = (int(e), k, None)
                sample = read_window(descriptor)
                data = preprocess(descriptor, sample) if sample is not None else None
                if data is not None:
                    yield data
        return

    logical_worker = global_worker if logical_worker is None else logical_worker
    saved = dataset.worker_resume_states.get(logical_worker) if dataset.resume_enabled else None
    cursor = dict(round_idx=0, episode_pos=0, frame=0, attempted=0, usable=0)
    if saved:
        cursor.update({key: int(saved["source"][key]) for key in cursor})
    queue = list(saved["queue"]) if saved else []
    buffer = [None] * len(queue)
    rng = np.random.default_rng([dataset.seed, global_worker, 1])
    if saved:
        rng.bit_generator.state = deepcopy(saved["shuffle_rng"])
    state = {"worker_id": logical_worker, "queue": queue}
    delivered = int(saved["delivered"]) if saved else 0
    for i in sorted(range(len(queue)), key=lambda i: queue[i][:2]):
        buffer[i] = read_window(queue[i])
        if buffer[i] is None:
            raise RuntimeError("checkpoint resident is no longer readable; dataset changed")

    def source():
        restoring = saved is not None
        while True:
            order = np.random.default_rng(
                [dataset.seed, cursor["round_idx"], global_worker, 2]
            ).permutation(assigned)
            drop_rng = np.random.default_rng([dataset.seed, cursor["round_idx"], global_worker, 0])
            if restoring:
                drop_rng.bit_generator.state = deepcopy(saved["source"]["drop_rng"])
                restoring = False
            while cursor["episode_pos"] < len(order):
                e = int(order[cursor["episode_pos"]])
                if cursor["frame"] == dataset.num_anchors(e):
                    cursor["episode_pos"] += 1
                    cursor["frame"] = 0
                    continue
                k = cursor["frame"]
                cursor["frame"] += 1
                if drop_rng.random() < dataset.drop_ratio:
                    continue
                cursor["attempted"] += 1
                seed = int(
                    np.random.SeedSequence(
                        [dataset.seed, cursor["round_idx"], global_worker, e, k, 3]
                    ).generate_state(1)[0]
                )
                descriptor = (e, k, seed)
                sample = read_window(descriptor)
                if sample is not None:
                    cursor["usable"] += 1
                    state["source"] = {**cursor, "drop_rng": drop_rng.bit_generator.state}
                    yield descriptor, sample
            if cursor["attempted"] and not cursor["usable"]:
                raise ValueError("all attempted source windows in this worker round failed to read")
            cursor["round_idx"] += 1
            cursor.update(episode_pos=0, frame=0, attempted=0, usable=0)

    windows = source()
    while True:
        # Warm up, then grow by one per output until the buffer reaches capacity.
        for _ in range(2 if len(buffer) < dataset.shuffle_buffer - 1 else 1):
            descriptor, sample = next(windows)
            queue.append(descriptor)
            buffer.append(sample)
        if len(buffer) < dataset.shuffle_initial:
            continue
        pick = int(rng.integers(len(buffer)))
        data = preprocess(queue[pick], buffer[pick])
        for items in (queue, buffer):
            items[pick] = items[-1]
            items.pop()
        if data is None:
            continue
        if dataset.resume_enabled:
            delivered += 1
            state.update(shuffle_rng=rng.bit_generator.state, delivered=delivered)
            data = StreamSample(data)
            data.stream_state, data.stream_name = state, dataset.stream_name
        yield data


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
    description = (
        dataset.resume_description()
        if hasattr(dataset, "resume_description")
        else {
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
    )
    description["collator"] = collator_config
    digest = hashlib.sha256(json.dumps(_json_value(description), sort_keys=True).encode())
    if getattr(dataset, "normalizer", None) is not None:
        for name, tensor in sorted(dataset.normalizer.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(
                tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


class StreamCheckpoint:
    """Track consumed batches for one stream or a fixed-ratio VLA/VLM pair."""

    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
        rank=0,
        world_size=1,
        micro_batches_per_epoch=None,
        collator_config=None,
        gradient_accumulation_steps=1,
    ):
        if (
            batch_size < 1
            or num_workers < 0
            or not 0 <= rank < world_size
            or gradient_accumulation_steps < 1
            or (micro_batches_per_epoch is not None and micro_batches_per_epoch < 1)
        ):
            raise ValueError("invalid stream loader topology")
        self.mixed = getattr(dataset, "vlm_dataset", None) is not None
        self.batch_size = int(batch_size)
        if self.mixed:
            self.vla_ratio = float(dataset.vla_ratio)
            vla_count = math.ceil(batch_size * self.vla_ratio)
            if not 0 < vla_count < batch_size or dataset.batch_size != batch_size:
                raise ValueError(
                    "mixed dataset requires matching batch size and nonempty VLA/VLM quotas"
                )
            self.datasets = {"vla": dataset.vla_dataset, "vlm": dataset.vlm_dataset}
            sizes = {"vla": vla_count, "vlm": batch_size - vla_count}
        else:
            self.datasets = {dataset.stream_name: dataset}
            sizes = {dataset.stream_name: batch_size}
        self.rank_key = f"rank_{int(rank)}"
        self.consumed_batches = 0
        self.specs, self.workers = {}, {}
        for name, stream in self.datasets.items():
            self.specs[name] = {
                "version": 3,
                "media_encoding": {"rgb": "jpeg", "quality": JPEG_QUALITY, "depth": "npy"},
                "batch_size": int(sizes[name]),
                "num_workers": int(num_workers),
                "rank": int(rank),
                "world_size": int(world_size),
                "seed": stream.seed,
                "drop_ratio": stream.drop_ratio,
                "shuffle_buffer": stream.shuffle_buffer,
                "shuffle_initial": stream.shuffle_initial,
                "micro_batches_per_epoch": micro_batches_per_epoch,
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "dataset_fingerprint": dataset_fingerprint(stream, collator_config),
            }
            self.workers[name] = {}
            stream.resume_enabled = True
            stream.resume_num_workers = max(1, num_workers)
            stream.resume_rank, stream.resume_world_size = int(rank), int(world_size)
            stream.resume_batches, stream.worker_resume_states = 0, {}
        self.spec = next(iter(self.specs.values()))

    def record_consumed(self, encoded):
        state = pickle.loads(encoded)
        states = state.get("streams", {}) if self.mixed else {next(iter(self.datasets)): state}
        if set(states) != set(self.datasets):
            raise ValueError("batch stream names disagree with the configured datasets")
        nw = max(1, self.spec["num_workers"])
        worker = self.consumed_batches % nw
        for name, item in states.items():
            expected = (self.consumed_batches // nw + 1) * self.specs[name]["batch_size"]
            if (
                item["worker_id"] != worker
                or state["worker_id"] != worker
                or item["delivered"] != expected
            ):
                raise ValueError("stream delivery count/order disagrees with consumed batches")
        for name, item in states.items():
            self.workers[name][worker] = item
        self.consumed_batches += 1

    def state_dict(self):
        streams = {}
        for name, spec in self.specs.items():
            payload = {
                "spec": spec,
                "consumed_batches": self.consumed_batches,
                "workers": self.workers[name],
            }
            streams[name] = {self.rank_key: pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)}
        if not self.mixed:
            return next(iter(streams.values()))
        payload = {
            "format": "vla_vlm_v1",
            "batch_size": self.batch_size,
            "vla_ratio": self.vla_ratio,
            "streams": streams,
        }
        return {self.rank_key: pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)}

    def load_state_dict(self, state):
        payload = pickle.loads(state[self.rank_key])
        if self.mixed:
            if (
                payload.get("format") != "vla_vlm_v1"
                or payload["batch_size"] != self.batch_size
                or payload["vla_ratio"] != self.vla_ratio
            ):
                raise ValueError("mixed stream checkpoint format or VLA/VLM ratio mismatch")
            payloads = {
                name: pickle.loads(item[self.rank_key]) for name, item in payload["streams"].items()
            }
        else:
            payloads = {next(iter(self.datasets)): payload}
        if set(payloads) != set(self.datasets):
            raise ValueError("checkpoint stream names disagree with the configured datasets")
        counts = {int(item["consumed_batches"]) for item in payloads.values()}
        if len(counts) != 1 or min(counts) < 0:
            raise ValueError("checkpoint consumption boundaries disagree or are negative")
        consumed = counts.pop()
        nw = max(1, self.spec["num_workers"])
        for name, item in payloads.items():
            spec, workers = self.specs[name], item["workers"]
            if item["spec"] != spec:
                changed = [key for key in spec if item["spec"].get(key) != spec[key]]
                raise ValueError(f"stream checkpoint mismatch: {', '.join(changed)}")
            if set(workers) != set(range(min(nw, consumed))):
                raise ValueError("checkpoint worker set disagrees with consumed batches")
            for wid, worker in workers.items():
                expected = ((consumed + nw - 1 - wid) // nw) * spec["batch_size"]
                if worker["worker_id"] != wid or worker["delivered"] != expected:
                    raise ValueError("invalid worker delivery count in stream checkpoint")
                if len(worker["queue"]) > spec["shuffle_buffer"] - 1:
                    raise ValueError("checkpoint shuffle queue exceeds capacity")
        self.consumed_batches = consumed
        for name, item in payloads.items():
            self.workers[name] = item["workers"]
            self.datasets[name].resume_batches = consumed
            self.datasets[name].worker_resume_states = item["workers"]

    def require_checkpoint(self, checkpoint_path):
        from torch.distributed.checkpoint import FileSystemReader

        keys = FileSystemReader(checkpoint_path).read_metadata().state_dict_metadata
        if f"app.data_stream.{self.rank_key}" not in keys:
            raise ValueError(
                "checkpoint has no LeRobot stream state for this rank; use finetune_checkpoint_path for weights only"
            )


class StreamCheckpointCollator:
    """Freeze worker state before the next prefetch can mutate the live queue."""

    def __init__(self, collator, stream_names=None):
        self.collator = collator
        self.stream_names = stream_names

    def __call__(self, samples):
        states = [sample.stream_state for sample in samples]
        if self.stream_names is not None:
            streams = {sample.stream_name: sample.stream_state for sample in samples}
            state = {"worker_id": states[-1]["worker_id"], "streams": streams}
        else:
            state = states[-1]
        result = self.collator(samples)
        result[STREAM_STATE_KEY] = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return result


class LeRobotDataset(torch.utils.data.IterableDataset):
    """Common lifecycle; datasets define num_anchors, read_window and sample_to_data."""

    lowdim_only = False
    stream_name = "vla"

    reader_type = LeRobotEpisodeReader

    def __init__(
        self,
        root,
        *,
        split,
        val_split,
        seed,
        mode,
        drop_ratio,
        shuffle_buffer,
        shuffle_initial,
        val_stride,
        reader_kwargs,
        target_image_size,
        sanity_checks,
        return_dataset_info,
    ):
        super().__init__()
        if (
            mode not in ("train", "val")
            or not 0 <= drop_ratio < 1
            or val_stride < 1
            or shuffle_buffer < 1
            or (shuffle_initial is not None and shuffle_initial < 1)
        ):
            raise ValueError("invalid stream sampling configuration")
        self.root, self.split, self.val_split = str(root), split, val_split
        self.mode, self.seed = mode, int(seed)
        self.drop_ratio, self.val_stride = float(drop_ratio), int(val_stride)
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_initial = min(shuffle_buffer, shuffle_initial or shuffle_buffer)
        self.target_image_size = tuple(target_image_size) if target_image_size is not None else None
        self.return_dataset_info = return_dataset_info
        self.sanity_checks = dict(sanity_checks or {})
        self.checker = DataChecker(sanity_cfg=self.sanity_checks)
        self.collator = None
        self.reader_kwargs = dict(reader_kwargs or {})
        self.reader = self.reader_type(root, split, **self.reader_kwargs)
        self.resume_enabled = False
        self.resume_batches = self.resume_rank = 0
        self.resume_num_workers = self.resume_world_size = 1
        self.worker_resume_states = {}

    def build_pipeline(self):
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
            if dist.is_available() and dist.is_initialized():
                rank, world = dist.get_rank(), dist.get_world_size()
            else:
                rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
            if world <= 0 or not 0 <= rank < world:
                raise ValueError(f"invalid rank/world size: {rank}/{world}")
            global_worker, total_workers = rank * nw + wid, world * nw
        return build_lerobot_pipeline(self, global_worker, total_workers, logical)

    def __iter__(self):
        return iter(self.build_pipeline())

    def get_validation_dataset(self):
        dataset = deepcopy(self)
        dataset.mode, dataset.split = "val", self.val_split
        if hasattr(dataset, "aug_transform"):
            dataset.aug_transform = False
        dataset.resume_enabled = False
        dataset.worker_resume_states = {}
        dataset.reader = self.reader_type(self.root, self.val_split, **self.reader_kwargs)
        return dataset

    def set_collator(self, collator):
        self.collator = collator

    def get_collator(self):
        assert self.collator is not None, "Collator not set"
        collator = UnifiedVLACollator(
            formatter=self.collator.formatter,
            batch_processor=deepcopy(self.collator.batch_processor),
            mode=self.mode,
            debug_capture_texts=self.collator.debug_capture_texts,
            debug_profile_timing=self.collator.debug_profile_timing,
        )
        if self.mode == "train" and self.resume_enabled:
            return StreamCheckpointCollator(collator)
        return collator
