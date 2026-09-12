"""Resumable episode/shuffle schedule; checkpoints contain descriptors, not RGB."""

import random
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch

from ..data_transforms import COLOR_AUG


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
