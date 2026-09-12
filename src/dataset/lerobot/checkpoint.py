"""Track learner-consumed data batches and store each rank in a distinct DCP key."""

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

import numpy as np
from omegaconf import OmegaConf
import torch


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
