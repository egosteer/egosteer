"""LeRobot/Parquet VLM samples using the existing WDS question-answer contract."""

import io
import json
from copy import deepcopy

import numpy as np
from PIL import Image

from ..sanity_checks import MissingOrInvalidFilesError
from ..wds.vlm_dataset import VLMWdsDataset
from .lerobot_dataset import LeRobotEpisodeReader, LeRobotStreamMixin


QA_FIELDS = ("texts", "formatting_ratings", "visual_dependency_ratings", "relevance_ratings")
QA_INFO_FIELDS = ("source", "dataset_name", "sample_idx")


class LeRobotVLMReader(LeRobotEpisodeReader):
    """One frame row is one VLM sample; no motion, calibration or next-state fields."""

    def __init__(self, root, split, image_keys=None, metadata_key=None, **kwargs):
        self.image_keys = list(image_keys) if image_keys is not None else None
        self.metadata_key = metadata_key
        super().__init__(root, split, **kwargs)

    def _validate_features(self):
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("expected codebase_version v3.0")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("info.json fps must be positive")
        features = self.info["features"]
        if self.image_keys is None:
            self.image_keys = sorted(
                key for key, feature in features.items()
                if feature["dtype"] in ("image", "video")
                and not ((feature.get("info") or {}).get("is_depth_map", False)
                         or (feature.get("info") or {}).get("video.is_depth_map", False)
                         or (feature.get("video_info") or {}).get("video.is_depth_map", False))
            )
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("VLM requires a nonempty, unique image_keys list")
        for key in self.image_keys:
            if key not in features or features[key]["dtype"] not in ("image", "video"):
                raise ValueError(f"VLM visual feature {key!r} must have dtype image or video")
        self.video_keys = tuple(key for key in self.image_keys if features[key]["dtype"] == "video")
        self.image_columns = [key for key in self.image_keys if key not in self.video_keys]
        self.frame_columns = (
            "episode_index", "frame_index", "index", "timestamp", *self.image_columns,
            *((self.metadata_key,) if self.metadata_key else QA_FIELDS),
        )
        self.episode_columns = {
            "episode_index", "length", "dataset_from_index", "dataset_to_index",
            "data/chunk_index", "data/file_index",
            *(f"videos/{key}/{field}" for key in self.video_keys
              for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")),
        }

    def _read_tasks(self):
        # QA text comes from each frame, not the robot task vocabulary.
        return {}

    def _validate_episode(self, row):
        self._validate_episode_bounds(row)
        keys = [key for key in self.video_keys if row.get(f"videos/{key}/file_index") is not None]
        self._validate_video_intervals(row, keys)

    def _read_image(self, value):
        """Decode a LeRobot/HF image feature: embedded bytes or root-relative path."""
        if isinstance(value, dict):
            value = value.get("bytes") if value.get("bytes") is not None else value.get("path")
        if value is None:
            return None
        if isinstance(value, (bytes, str)):
            source = io.BytesIO(value) if isinstance(value, bytes) else self.root / value
            with Image.open(source) as image:
                return image.convert("RGB")
        raise MissingOrInvalidFilesError("VLM image must contain bytes or a path")

    def read_sample(self, e, k, load_media=True):
        path = str(self.data_path(self.episodes[e]))
        available = set(self._get_file(path).schema_arrow.names)
        columns = [self.metadata_key] if self.metadata_key else list(QA_FIELDS)
        columns += [key for key in QA_INFO_FIELDS if key in available]
        if load_media:
            columns += self.image_columns
        row = self.read_rows(e, [k], columns)[k]
        meta = row[self.metadata_key] if self.metadata_key else {key: row[key] for key in QA_FIELDS}
        if isinstance(meta, str):
            meta = json.loads(meta)
        if not isinstance(meta, dict):
            raise MissingOrInvalidFilesError("VLM metadata must be a mapping")
        meta = dict(meta)
        for key in QA_FIELDS:
            if key not in meta:
                raise MissingOrInvalidFilesError(f"VLM metadata missing {key}")
            if isinstance(meta[key], str):
                meta[key] = json.loads(meta[key])
        texts = meta["texts"]
        if (not isinstance(texts, list) or not texts
                or not all(isinstance(turn, dict) and isinstance(turn.get("user"), str)
                           and isinstance(turn.get("assistant"), str) for turn in texts)):
            raise MissingOrInvalidFilesError("texts must contain user/assistant question-answer pairs")
        for key in QA_FIELDS[1:]:
            ratings = meta[key]
            if not isinstance(ratings, list) or len(ratings) != len(texts):
                raise MissingOrInvalidFilesError(f"{key} must have one entry per candidate QA pair")
            if not all(value is None or (isinstance(value, (int, float)) and np.isfinite(value)) for value in ratings):
                raise MissingOrInvalidFilesError(f"{key} must contain finite ratings or null")
        meta.update({key: row[key] for key in QA_INFO_FIELDS if key in row})
        meta.setdefault("source", self.root.name)
        meta.setdefault("sample_idx", row["index"])
        sample = {"meta.json": meta, "__key__": f"episode_{self.episodes[e]['episode_index']}_frame_{k}"}
        if load_media:
            for i, key in enumerate(self.image_keys):
                if key in self.video_keys:
                    if self.episodes[e].get(f"videos/{key}/file_index") is None:
                        continue
                    image = Image.fromarray(self.read_media(e, key, [k])[0])
                elif row[key] is not None:
                    image = self._read_image(row[key])
                else:
                    continue
                if image is None:
                    continue
                # Preserve configured visual order when the WDS adapter sorts keys.
                sample[f"image_{i:06d}.jpg"] = image
        return sample


class VLMLeRobotDataset(LeRobotStreamMixin, VLMWdsDataset):
    """Stream native VLM rows; reuse WDS QA selection, image transforms and collation."""

    stream_name = "vlm"

    def __init__(self, root, split="train", val_split="val", weights=None, seed=42,
                 mode="train", target_image_size=None, shuffle_buffer=4096,
                 shuffle_initial=4096, drop_ratio=0.0, val_stride=1, image_keys=None,
                 metadata_key=None, reader_kwargs=None, sanity_checks=None, return_dataset_info=False):
        if not 0 <= drop_ratio < 1 or seed < 0 or val_stride < 1 or shuffle_buffer < 1:
            raise ValueError("invalid VLM sampling configuration")
        if shuffle_initial is not None and shuffle_initial < 1:
            raise ValueError("shuffle_initial must be positive")
        if mode not in ("train", "val"):
            raise ValueError("mode must be train or val")
        weights = [0.5, 0.5, 0.5] if weights is None else list(weights)
        if len(weights) != 3 or not np.isfinite(weights).all():
            raise ValueError("VLM QA scoring requires three finite weights")
        super().__init__(
            wds_datasets=[], weights=weights, seed=seed, mode=mode,
            shuffle_buffer=shuffle_buffer, shuffle_initial=shuffle_initial,
            target_image_size=target_image_size, keep_ratio=1 - drop_ratio,
            sanity_checks=sanity_checks, return_dataset_info=return_dataset_info,
        )
        self.root, self.split, self.val_split = str(root), split, val_split
        self.drop_ratio, self.val_stride = float(drop_ratio), int(val_stride)
        self.shuffle_initial = min(shuffle_buffer, shuffle_initial or shuffle_buffer)
        self.reader_kwargs = {**(reader_kwargs or {}), "image_keys": image_keys, "metadata_key": metadata_key}
        self.reader = LeRobotVLMReader(root, split, **self.reader_kwargs)
        self.resume_enabled = False
        self.resume_batches = 0
        self.resume_num_workers = 1
        self.resume_rank = 0
        self.resume_world_size = 1
        self.worker_resume_states = {}
        if mode == "train" and not len(self):
            raise ValueError("VLM training split has no samples")

    def __len__(self):
        return sum(self.num_anchors(e) for e in range(len(self.reader.episodes)))

    def num_anchors(self, e):
        # Unlike VLA, a VLM frame does not require a successor.
        return int(self.reader.episodes[e]["length"])

    def read_window(self, e, k, load_media=True):
        return self.reader.read_sample(e, k, load_media=load_media)

    def resume_description(self):
        return {
            "kind": "vlm", "info": self.reader.info, "episodes": self.reader.episodes,
            "split": self.split, "weights": self.weights, "image_keys": self.reader.image_keys,
            "metadata_key": self.reader.metadata_key, "target_image_size": self.target_image_size,
            "sanity_checks": self.sanity_checks,
        }

    def get_validation_dataset(self):
        if self.val_split is None:
            raise ValueError("VLM val_split is not configured")
        dataset = deepcopy(self)
        dataset.mode, dataset.split = "val", self.val_split
        dataset.resume_enabled = False
        dataset.worker_resume_states = {}
        dataset.reader = LeRobotVLMReader(self.root, self.val_split, **self.reader_kwargs)
        return dataset
