from __future__ import annotations

import pickle
import random
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    get_model_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


APP_STATE_KEY = "app"


def capture_rng_state() -> dict[str, Any]:
    """Capture this rank's main-process RNGs without initializing CUDA devices."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


@contextmanager
def preserve_rng_state(enabled=True):
    """Keep checkpoint I/O and logging from perturbing the learner's RNGs."""
    state = capture_rng_state() if enabled else None
    try:
        yield state
    finally:
        if state is not None:
            restore_rng_state(state)


class FSDPWorkspaceAppState(Stateful):
    """Unified DCP app state for FSDP2 training checkpoints."""

    def __init__(
        self,
        *,
        model,
        optimizer,
        lr_scheduler,
        training_state,
        data_stream=None,
        rng_state=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_state = training_state
        self.data_stream = data_stream
        self.rng_state = rng_state
        self._options = StateDictOptions(strict=True)

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optim_state_dict = get_state_dict(
            self.model,
            self.optimizer,
            options=self._options,
        )
        state = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "training_state": self.training_state.state_dict(),
        }
        if self.data_stream is not None:
            state["data_stream"] = self.data_stream.state_dict()
        if self.rng_state is not None:
            state["rng_state"] = {
                self.data_stream.rank_key: pickle.dumps(
                    self.rng_state, protocol=pickle.HIGHEST_PROTOCOL
                )
            }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        incompatible_keys = set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=self._options,
        )
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "FSDP checkpoint restore reported incompatible keys: "
                f"missing={incompatible_keys.missing_keys}, "
                f"unexpected={incompatible_keys.unexpected_keys}"
            )

        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.training_state.load_state_dict(state_dict["training_state"])
        if self.data_stream is not None:
            self.data_stream.load_state_dict(state_dict["data_stream"])
        if self.rng_state is not None:
            # Setup/compile and DataLoader startup still follow. Apply immediately before train_step.
            self.rng_state = pickle.loads(state_dict["rng_state"][self.data_stream.rank_key])


class FSDPModelOnlyAppState(Stateful):
    """DCP app state for loading only the model weights from a workspace checkpoint."""

    def __init__(self, *, model) -> None:
        self.model = model
        self._options = StateDictOptions(strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": get_model_state_dict(
                self.model,
                options=self._options,
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        incompatible_keys = set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self._options,
        )
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            raise RuntimeError(
                "FSDP model-only checkpoint restore reported incompatible keys: "
                f"missing={incompatible_keys.missing_keys}, "
                f"unexpected={incompatible_keys.unexpected_keys}"
            )
