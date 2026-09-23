"""Shared VLA/VLM stream interleaving and validation batching."""

import math

import torch


class UnifiedDataset(torch.utils.data.IterableDataset):
    """Shared wrapper for VLA streams and optional VLM interleaving.

    Training mode: VLM samples interleaved at a fixed ratio, VLM auto-restarts.
    Validation mode: all VLA samples first, then all VLM samples sequentially.
    """

    def __init__(
        self,
        vla_dataset,
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
        has_vlm_val = (
            getattr(self.vlm_dataset, "val_wds_datasets", None) is not None
            or getattr(self.vlm_dataset, "val_split", None) is not None
        )
        if (
            self.vlm_dataset is not None
            and has_vlm_val
            and hasattr(self.vlm_dataset, 'get_validation_dataset')
        ):
            vlm_val = self.vlm_dataset.get_validation_dataset()

        return type(self)(
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
