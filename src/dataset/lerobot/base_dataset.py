"""Shared worker partitioning, preprocessing and resume for LeRobot streams."""

import os
import time

import numpy as np
import torch
import torch.distributed as dist

from src.utils.pytorch_util import dict_apply
from ..sanity_checks import DataSkipError, current_worker_id
from .stream import ResumableEpisodeStream


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
            from .checkpoint import StreamCheckpointCollator
            return StreamCheckpointCollator(collator)
        return collator
