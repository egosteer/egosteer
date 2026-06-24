from __future__ import annotations

from typing import Any

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    get_model_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


APP_STATE_KEY = "app"


class FSDPWorkspaceAppState(Stateful):
    """Unified DCP app state for FSDP2 training checkpoints."""

    def __init__(
        self,
        *,
        model,
        optimizer,
        lr_scheduler,
        training_state,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_state = training_state
        self._options = StateDictOptions(strict=True)

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optim_state_dict = get_state_dict(
            self.model,
            self.optimizer,
            options=self._options,
        )
        return {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "training_state": self.training_state.state_dict(),
        }

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
