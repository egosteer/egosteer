"""Projected Parquet + per-feature video reader for the released dataset."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import CALIBRATION_SHAPES, EPISODE_COLUMNS, FRAME_COLUMNS, VIDEO_KEYS
from .video import VideoFrameCache, depth_parameters


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
