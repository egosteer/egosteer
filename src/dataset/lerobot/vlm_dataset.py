"""LeRobot/Parquet VLM samples using the existing WDS question-answer contract."""

import io
import json
from copy import deepcopy

import numpy as np
from PIL import Image

from ..sanity_checks import MissingOrInvalidFilesError
from ..data_transforms import process_image
from .lerobot_dataset import encode_frame, LeRobotEpisodeReader, LeRobotDataset


QA_FIELDS = ("texts", "formatting_ratings", "visual_dependency_ratings", "relevance_ratings")
QA_INFO_FIELDS = ("source", "dataset_name", "sample_idx")


class LeRobotVLMReader(LeRobotEpisodeReader):
    """One frame row is one VLM sample; no motion, calibration or next-state fields."""

    is_vla = False

    def __init__(self, root, split, image_keys=None, metadata_key=None, **kwargs):
        self.image_keys = list(image_keys) if image_keys is not None else None
        self.metadata_key = metadata_key
        super().__init__(root, split, **kwargs)

    # Discover image/video columns independently of the VLA motion and calibration schema.
    def _validate_features(self):
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("info.json fps must be positive")
        features = self.info["features"]
        if self.image_keys is None:
            self.image_keys = sorted(
                key
                for key, feature in features.items()
                if feature["dtype"] in ("image", "video")
                and not (
                    (feature.get("info") or {}).get("is_depth_map", False)
                    or (feature.get("info") or {}).get("video.is_depth_map", False)
                    or (feature.get("video_info") or {}).get("video.is_depth_map", False)
                )
            )
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("VLM requires a nonempty, unique image_keys list")
        for key in self.image_keys:
            if key not in features or features[key]["dtype"] not in ("image", "video"):
                raise ValueError(f"VLM visual feature {key!r} must have dtype image or video")
        self.video_keys = tuple(
            key for key in self.image_keys if features[key]["dtype"] == "video"
        )
        self.image_columns = [key for key in self.image_keys if key not in self.video_keys]
        self.episode_columns = {
            "episode_index",
            "length",
            "dataset_from_index",
            "dataset_to_index",
            "data/chunk_index",
            "data/file_index",
            *(
                f"videos/{key}/{field}"
                for key in self.video_keys
                for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            ),
        }

    # Accept the official image representation: embedded bytes or a dataset-relative file path.
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

    # One Parquet row becomes one QA sample; preserve visual order and defer QA selection until dequeue.
    def read_sample(self, e, k, load_media=True, compressed=False, target_size=None):
        available = self.columns[self.data_paths[e]]
        columns = [self.metadata_key] if self.metadata_key else list(QA_FIELDS)
        columns += [key for key in QA_INFO_FIELDS if key in available]
        if load_media:
            columns += self.image_columns
        row = self.read_table(e, [k], columns).to_pylist()[0]
        meta = (
            row[self.metadata_key]
            if self.metadata_key
            else {key: row[key] for key in QA_FIELDS}
        )
        if isinstance(meta, str):
            meta = json.loads(meta)
        if not isinstance(meta, dict):
            raise MissingOrInvalidFilesError("VLM metadata must be a mapping")
        meta = dict(meta)
        meta.update({key: row[key] for key in QA_INFO_FIELDS if key in row})
        meta.setdefault("source", self.root.name)
        meta.setdefault("sample_idx", row["index"])
        sample = {
            "meta.json": meta,
            "__key__": f"episode_{self.episodes[e]['episode_index']}_frame_{k}",
        }
        if load_media:
            for i, key in enumerate(self.image_keys):
                if key in self.video_keys:
                    if self.episodes[e].get(f"videos/{key}/file_index") is None:
                        continue
                    frames = self.read_media(
                        e, key, [k], compressed=compressed, target_size=target_size
                    )
                    if compressed:
                        image = frames[0]
                    else:
                        image = Image.fromarray(frames[0])
                elif row[key] is not None:
                    image = self._read_image(row[key])
                    if compressed and image is not None:
                        image = encode_frame(np.asarray(image), target_size)
                else:
                    continue
                if image is None:
                    continue
                # Preserve configured visual order when the WDS adapter sorts keys.
                sample[f"image_{i:06d}.jpg"] = image
        return sample


class VLMLeRobotDataset(LeRobotDataset):
    """Stream native VLM rows; select QA and apply the standard model preprocessing locally."""

    stream_name = "vlm"
    reader_type = LeRobotVLMReader

    # Reuse the episode stream and resume machinery; weights here score QA candidates, not data sources.
    def __init__(
        self,
        root,
        split="train",
        val_split="val",
        weights=None,
        seed=42,
        mode="train",
        target_image_size=None,
        shuffle_buffer=16384,
        shuffle_initial=4096,
        drop_ratio=0.0,
        val_stride=1,
        image_keys=None,
        metadata_key=None,
        reader_kwargs=None,
        sanity_checks=None,
        return_dataset_info=False,
        resume_warmup_samples=4096,
    ):
        weights = [0.5, 0.5, 0.5] if weights is None else list(weights)
        if len(weights) != 3 or not np.isfinite(weights).all():
            raise ValueError("VLM QA scoring requires three finite weights")
        reader_kwargs = {
            **(reader_kwargs or {}),
            "image_keys": image_keys,
            "metadata_key": metadata_key,
        }
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
        self.weights = weights
        if mode == "train" and not len(self):
            raise ValueError("VLM training split has no samples")

    def __len__(self):
        return sum(self.num_anchors(e) for e in range(len(self.episodes)))

    def num_anchors(self, e):
        # Unlike VLA, a VLM frame does not require a successor.
        return int(self.episodes[e]["length"])

    # Route the global episode descriptor to its source; VLM includes the final frame.
    def read_window(self, e, k, load_media=True, compressed=False):
        source_id, e = self.episode_sources[e]
        return self.readers[source_id].read_sample(
            e,
            k,
            load_media=load_media,
            compressed=compressed,
            target_size=self.target_image_size,
        )

    # Post-shuffle VLM stage: validate candidates, take the highest weighted score, then process images.
    def sample_to_data(self, sample):
        """Validate QA and select/augment after shuffle, as in the WDS pipeline."""
        sample = {**sample, "meta.json": dict(sample["meta.json"])}
        meta = sample["meta.json"]
        for key in QA_FIELDS:
            if key not in meta:
                raise MissingOrInvalidFilesError(f"VLM metadata missing {key}")
            if isinstance(meta[key], str):
                meta[key] = json.loads(meta[key])
        texts = meta["texts"]
        if (
            not isinstance(texts, list)
            or not texts
            or not all(
                isinstance(turn, dict)
                and isinstance(turn.get("user"), str)
                and isinstance(turn.get("assistant"), str)
                for turn in texts
            )
        ):
            raise MissingOrInvalidFilesError(
                "texts must contain user/assistant question-answer pairs"
            )
        for key in QA_FIELDS[1:]:
            ratings = meta[key]
            if not isinstance(ratings, list) or len(ratings) != len(texts):
                raise MissingOrInvalidFilesError(
                    f"{key} must have one entry per candidate QA pair"
                )
            if not all(
                value is None or (isinstance(value, (int, float)) and np.isfinite(value))
                for value in ratings
            ):
                raise MissingOrInvalidFilesError(f"{key} must contain finite ratings or null")
        self.checker.check(
            sample_schema=(
                sample,
                {
                    "required_keys": ("meta.json",),
                    "required_meta_keys": (
                        "texts",
                        "formatting_ratings",
                        "visual_dependency_ratings",
                        "relevance_ratings",
                    ),
                },
            )
        )

        image_keys = sorted(
            [k for k in sample.keys() if k.startswith("image_") and k.endswith(".jpg")]
        )
        if not image_keys:
            raise MissingOrInvalidFilesError("missing required image_*.jpg fields")
        images = [sample[k] for k in image_keys]

        text = meta["texts"]
        weights = self.weights

        if len(text) > 1:
            formatting_ratings, visual_dependency_ratings, relevance_ratings = (
                np.array([r if r is not None else 0 for r in meta[key]])
                for key in QA_FIELDS[1:]
            )
            scores = (
                formatting_ratings * weights[0]
                + visual_dependency_ratings * weights[1]
                + relevance_ratings * weights[2]
            )
            text = text[np.argmax(scores)]
        else:
            text = text[0]
        question = str(text["user"])
        answer = str(text["assistant"])
        self.checker.check(instruction=(question, 1))

        raw_images = []
        for img_pil in images:
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            raw_images.append(np.asarray(img_pil, dtype=np.uint8))
        images_arr = np.stack(raw_images, dtype=np.uint8)

        # Always resize: dynamic aspect ratios cause vision-tower recompiles.
        images_processed, _, _ = process_image(
            images_arr,
            aug_transform=(self.mode == "train"),
            target_size=self.target_image_size,
        )
        self.checker.check(finite={"images_processed": images_processed})

        data = {
            "images": images_processed,
            "question": question,
            "answer": answer,
            "vision_type": "image",
            "is_vla_data": np.array(False, dtype=bool),
            "view_mask": np.array([False, False], dtype=bool),
        }

        if self.return_dataset_info:
            data["dataset_name"] = meta.get("source", meta.get("dataset_name", "unknown"))
            data["episode_index"] = np.array(meta.get("sample_idx", -1), dtype=np.int32)
        self.checker.check(finite=data)
        return data

    # Include QA scoring and visual selection in the stream compatibility fingerprint.
    def resume_description(self):
        return {
            "kind": "vlm",
            "info": self.reader.info,
            "episodes": self.reader.episodes,
            "split": self.split,
            "weights": self.weights,
            "image_keys": self.reader.image_keys,
            "metadata_key": self.reader.metadata_key,
            "target_image_size": self.target_image_size,
            "sanity_checks": self.sanity_checks,
        }
