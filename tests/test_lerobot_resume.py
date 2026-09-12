"""Consumption-boundary resume, including real workers and rank-local DCP leaves."""

from copy import deepcopy
from datetime import timedelta
from itertools import islice
import pickle
import random
import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from test_lerobot_stream import release_root, dataset
from src.dataset.stream_checkpoint import StreamCheckpoint, StreamCheckpointCollator, STREAM_STATE_KEY
from src.dataset.vla_dataset import ViewDropoutConfig
from src.utils.fsdp_app_state import FSDPWorkspaceAppState


def simple_collate(samples):
    return {
        "instruction": [sample["instruction"] for sample in samples],
        "images": torch.stack([sample["images"] for sample in samples]),
        "chest_images": torch.stack([sample["chest_images"] for sample in samples]),
        "states": torch.stack([sample["states"] for sample in samples]),
        "actions": torch.stack([sample["actions"] for sample in samples]),
        "actions_valid_mask": torch.stack([sample["actions_valid_mask"] for sample in samples]),
        "view_mask": torch.stack([sample["view_mask"] for sample in samples]),
        "episode_index": [int(sample["episode_index"]) for sample in samples],
    }


def make_stream(root, workers=0, rank=0, world=1, **overrides):
    options = dict(mode="train", seed=37, drop_ratio=.2,
                   view_dropout=ViewDropoutConfig(drop_head=.2, drop_chest=.2))
    options.update(overrides)
    ds = dataset(root, **options)
    state = StreamCheckpoint(ds, batch_size=2, num_workers=workers, rank=rank, world_size=world,
                             micro_batches_per_epoch=3, collator_config={"test": True})
    return ds, state


def loader_for(ds, workers=0):
    kwargs = {} if workers == 0 else {"multiprocessing_context": "spawn", "prefetch_factor": 4}
    return torch.utils.data.DataLoader(ds, batch_size=2, num_workers=workers,
                                      collate_fn=StreamCheckpointCollator(simple_collate), **kwargs)


def take(iterator, state=None):
    batch = next(iterator)
    encoded = batch.pop(STREAM_STATE_KEY)
    if state is not None:
        state.record_consumed(encoded)
    return batch


def equal_batch(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


def shutdown(iterator):
    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def test_resume_restores_samples_augmentations_and_cursor(release_root):
    ds, state = make_stream(release_root)
    it = iter(loader_for(ds))
    for _ in range(3):
        take(it, state)
    saved = state.state_dict()
    expected = [take(it) for _ in range(8)]
    # Polluting process RNG must not change queued or subsequently read samples.
    random.seed(998)
    np.random.seed(998)
    torch.manual_seed(998)
    restored_ds, restored = make_stream(release_root)
    restored.load_state_dict(saved)
    assert divmod(restored.consumed_batches, 3) == (1, 0)
    new_it = iter(loader_for(restored_ds))
    for batch in expected:
        equal_batch(take(new_it, restored), batch)
    assert restored.consumed_batches == 11
    assert len(saved[state.rank_key]) < 100_000  # descriptors, not image tensors


@pytest.mark.parametrize("consumed", [1, 3])
def test_prefetch_and_worker_rotation(release_root, consumed):
    ds, state = make_stream(release_root, workers=2)
    it = iter(loader_for(ds, workers=2))
    try:
        for _ in range(consumed):
            take(it, state)
        saved = state.state_dict()
        # Workers have already prefetched more; only consumed states were saved.
        expected = [take(it) for _ in range(5)]
        assert state.consumed_batches == consumed
    finally:
        shutdown(it)
    restored_ds, restored = make_stream(release_root, workers=2)
    restored.load_state_dict(saved)
    it = iter(loader_for(restored_ds, workers=2))
    try:
        for batch in expected:
            equal_batch(take(it, restored), batch)
    finally:
        shutdown(it)


def test_equal_4096_warmup_and_capacity(release_root):
    ds, state = make_stream(release_root, drop_ratio=0, shuffle_buffer=4096, shuffle_initial=4096)
    calls = []
    def cheap_sample(e, k):
        calls.append((e, k))
        return {"index": (e, k)}
    ds.materialize = cheap_sample
    it = iter(ds)
    first = next(it)
    assert len(calls) == 4096
    assert len(first.stream_state["queue"]) == 4095
    next(it)
    assert len(calls) == 4097
    assert len(first.stream_state["queue"]) == 4095


@pytest.mark.parametrize("change", ["workers", "seed", "buffer", "metadata", "normalizer", "collator"])
def test_changed_resume_contract_is_rejected(release_root, change):
    _, saved = make_stream(release_root)
    ds, restored = make_stream(release_root, workers=2 if change == "workers" else 0,
                               **({"seed": 38} if change == "seed" else {}),
                               **({"shuffle_buffer": 5} if change == "buffer" else {}))
    if change == "metadata":
        ds.reader.episodes[0]["instructions"] = ["changed instruction"]
    elif change == "normalizer":
        ds.normalizer = torch.nn.Linear(48, 48)
    if change in ("metadata", "normalizer", "collator"):
        restored = StreamCheckpoint(ds, 2, 0, micro_batches_per_epoch=3,
                                    collator_config={"test": change != "collator"})
    with pytest.raises(ValueError, match="stream checkpoint mismatch"):
        restored.load_state_dict(saved.state_dict())


def test_collator_snapshot_is_immutable_under_prefetch(release_root):
    ds, state = make_stream(release_root)
    it = iter(loader_for(ds))
    batch = next(it)
    frozen = batch[STREAM_STATE_KEY]
    before = pickle.loads(frozen)
    for _ in range(3):
        next(it)
    assert pickle.loads(frozen) == before
    state.record_consumed(frozen)
    assert state.consumed_batches == 1


def test_cursor_resume_preserves_skipped_sample_positions(release_root):
    def make_skipping():
        ds, state = make_stream(release_root, drop_ratio=0)
        materialize = ds.materialize
        ds.materialize = lambda e, k: None if k % 2 else materialize(e, k)
        return ds, state
    ds, state = make_skipping()
    it = iter(loader_for(ds))
    for _ in range(2):
        take(it, state)
    saved = state.state_dict()
    expected = [take(it) for _ in range(5)]
    ds, state = make_skipping()
    state.load_state_dict(saved)
    it = iter(loader_for(ds))
    for batch in expected:
        equal_batch(take(it, state), batch)


class TinyTrainingState:
    def __init__(self):
        self.steps = 5

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = state["steps"]


def dcp_rank_worker(rank, port, root, path):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}?use_libuv=0", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        # Each rank spawns a DataLoader worker; rank ownership must survive even
        # without torchrun RANK/WORLD_SIZE environment variables in that worker.
        ds, state = make_stream(root, workers=1, rank=rank, world=2)
        it = iter(loader_for(ds, workers=1))
        for _ in range(rank + 2):
            take(it, state)
        torch.manual_seed(123)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        app = FSDPWorkspaceAppState(model=model, optimizer=optimizer, lr_scheduler=scheduler,
                                    training_state=TinyTrainingState(), data_stream=state)
        dcp.save({"app": app}, checkpoint_id=path)
        expected = [take(it) for _ in range(4)]
        shutdown(it)
        new_ds, new_state = make_stream(root, workers=1, rank=rank, world=2)
        new_state.require_checkpoint(path)
        app.data_stream = new_state
        dcp.load({"app": app}, checkpoint_id=path)
        assert new_state.consumed_batches == rank + 2
        new_it = iter(loader_for(new_ds, workers=1))
        for batch in expected:
            equal_batch(take(new_it, new_state), batch)
        shutdown(new_it)
    finally:
        dist.destroy_process_group()


def test_two_rank_dcp_round_trip(release_root, tmp_path):
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    mp.spawn(dcp_rank_worker, args=(port, str(release_root), str(tmp_path / "checkpoint")), nprocs=2, join=True)
    keys = dcp.FileSystemReader(tmp_path / "checkpoint").read_metadata().state_dict_metadata
    assert "app.data_stream.rank_0" in keys
    assert "app.data_stream.rank_1" in keys


def test_legacy_checkpoint_missing_stream_fails_explicitly(release_root, tmp_path):
    dcp.save({"app": {"training_state": {"global_step": 3}}}, checkpoint_id=str(tmp_path / "legacy"))
    _, state = make_stream(release_root)
    with pytest.raises(ValueError, match="no LeRobot stream state"):
        state.require_checkpoint(str(tmp_path / "legacy"))
