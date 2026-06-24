from typing import Optional, Dict
import logging
import os
import pathlib
import shutil
import sys
import types

import torch

from src.utils.fsdp_app_state import APP_STATE_KEY, FSDPModelOnlyAppState

log = logging.getLogger(__name__)


def enable_pathlib_local_pickle_compat() -> None:
    """Register a runtime shim for Python 3.13 pathlib pickle payloads.

    Python 3.13 stores concrete Path classes under pathlib._local, while
    Python 3.10 still exposes them from pathlib.py. Torch distributed
    checkpoint metadata is read with pickle.load, so checkpoints written in a
    newer environment can fail to deserialize in older runtimes unless that
    module path is mapped.

    Reference:
    - CPython 3.13 pathlib package layout: Lib/pathlib/_local.py
    - PyTorch FileSystemReader.read_metadata uses pickle.load on .metadata
    """
    if "pathlib._local" in sys.modules:
        return

    shim = types.ModuleType("pathlib._local")
    for name in [
        "Path",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
    ]:
        value = getattr(pathlib, name, None)
        if value is not None:
            setattr(shim, name, value)
    sys.modules["pathlib._local"] = shim

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:04d}-train_loss={train_loss:.4f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def propose_ckpt_path(
        self, data: Dict[str, float]
    ) -> "tuple[Optional[str], Optional[str], Optional[float]]":
        # Pure decision: no map mutation, no disk I/O. Caller must save
        # successfully (and verify) before calling commit(); a failed save
        # therefore cannot delete the displaced ckpt nor leave a ghost in
        # path_value_map.
        if self.k == 0:
            return None, None, None

        value = data[self.monitor_key]
        new_path = os.path.join(self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            return new_path, None, value

        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        if self.mode == 'max':
            delete_path = min_path if value > min_value else None
        else:
            delete_path = max_path if value < max_value else None

        if delete_path is None:
            return None, None, None
        return new_path, delete_path, value

    def commit(
        self,
        rank: int,
        new_path: str,
        value: float,
        delete_path: Optional[str],
    ) -> None:
        # Must be called only after the save to new_path is verified durable.
        if delete_path is not None:
            self.path_value_map.pop(delete_path, None)
        self.path_value_map[new_path] = value

        if rank == 0 and delete_path is not None and os.path.exists(delete_path):
            if os.path.isfile(delete_path):
                os.remove(delete_path)
            else:
                shutil.rmtree(delete_path)


def load_state_dict_checked(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    allow_missing_prefixes: tuple[str, ...] = (),
) -> None:
    """Controlled non-strict load.

    Tolerates only keys missing under known prefixes that the model rebuilds by
    itself at construction time (e.g. ``frozen_teacher.`` whose DINOv3 weights
    come from ``from_pretrained``). Any unexpected key, or any missing key
    outside the whitelist, raises instead of silently producing a partially
    loaded model. ``str.startswith`` accepts a tuple of prefixes directly.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        k for k in incompatible.missing_keys
        if not k.startswith(allow_missing_prefixes)
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint load mismatch. unexpected_keys={unexpected}; "
            f"missing_keys outside {allow_missing_prefixes}={missing}"
        )


def load_checkpoint(
    model: torch.nn.Module,
    path: str | pathlib.Path,
    *,
    allow_missing_prefixes: tuple[str, ...] = ("frozen_teacher.",),
) -> None:
    """Load model weights from a checkpoint file or Accelerate directory.

    Supports three formats:
    1. Single file (.pt / .ckpt) — torch.load with key probing.
    2. Native DCP training checkpoint dir (contains .metadata).
       Uses a model-only AppState over torch.distributed.checkpoint.
    3. Accelerate FSDP2 sharded dir (contains pytorch_model_fsdp_0/).
       Uses torch.distributed.checkpoint with no_dist=True (PyTorch 2.3+).
    4. Accelerate safetensors dir — falls back to load_checkpoint_in_model.

    The single-file path uses a controlled non-strict load (see
    ``load_state_dict_checked``): keys under ``allow_missing_prefixes`` may be
    absent — this is how released weights drop the frozen DINOv3 teacher, which
    the model rebuilds via ``from_pretrained`` — while any other missing or
    unexpected key still raises. The DCP / sharded / safetensors paths load full
    training checkpoints (teacher present) and stay strict.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Single-file checkpoint
    if path.is_file():
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        for key in ["model", "module", "model_state_dict"]:
            if key in state_dict:
                state_dict = state_dict[key]
                break
        load_state_dict_checked(model, state_dict, allow_missing_prefixes)
        log.info("Loaded single-file checkpoint from %s", path)
        return

    # Native DCP workspace checkpoint
    if (path / ".metadata").exists():
        import torch.distributed.checkpoint as dcp

        enable_pathlib_local_pickle_compat()
        app_state = FSDPModelOnlyAppState(model=model)
        dcp.load(
            state_dict={APP_STATE_KEY: app_state},
            storage_reader=dcp.FileSystemReader(str(path)),
            no_dist=True,
        )
        log.info("Loaded native DCP checkpoint from %s", path)
        return

    # Accelerate FSDP2 sharded checkpoint
    # dcp.load supports no_dist=True since PyTorch 2.3+ (fix: pytorch#115660),
    # and auto-infers it when dist is not initialized since 2.4+ (pytorch#118554).
    fsdp_dir = path / "pytorch_model_fsdp_0"
    if fsdp_dir.exists():
        import torch.distributed.checkpoint as dcp

        enable_pathlib_local_pickle_compat()
        state_dict = {"model": model.state_dict()}
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(str(fsdp_dir)),
            no_dist=True,
        )
        model.load_state_dict(state_dict["model"], strict=True)
        log.info("Loaded FSDP sharded checkpoint from %s", fsdp_dir)
        return

    # Safetensors / other Accelerate format
    # Source: accelerate.utils.load_checkpoint_in_model
    # load_checkpoint_in_model does not support strict mode natively;
    # load into a temporary state_dict and use strict load_state_dict instead.
    from safetensors.torch import load_file

    safetensor_files = sorted(path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")
    state_dict = {}
    for sf in safetensor_files:
        state_dict.update(load_file(str(sf), device="cpu"))
    model.load_state_dict(state_dict, strict=True)
    log.info("Loaded safetensors checkpoint from %s (%d files)", path, len(safetensor_files))


def load_model_and_collator_from_saved_config(
    train_config_path: str | pathlib.Path,
    device: str,
    collator_mode: str = "train",
    dtype: torch.dtype = torch.bfloat16,
):
    """Instantiate model + collator from a training run's saved .hydra/config.yaml.

    Mirrors the loading path in ``evaluate.py::main`` (lines 119-143): load
    the training config, instantiate ``cfg.policy``, move to device in the
    target dtype, and instantiate the collator. Use this when the checkpoint
    was trained with options (use_kv_projection, world_model,
    intermediate_size, ...) that differ from the current code's default
    experiment config.

    Note: this function does NOT register OmegaConf resolvers. Callers that
    need ``${eval:...}`` interpolations in saved configs must register the
    resolver at module import time via::

        OmegaConf.register_new_resolver("eval", eval, replace=True)

    Args:
        train_config_path: Path to the run's saved ``.hydra/config.yaml``.
        device: Target device string (e.g. ``"cuda"``).
        collator_mode: Mode passed to the collator constructor (default
            ``"train"`` to match the most common analysis use case).
        dtype: Tensor dtype for ``model.to(...)`` (default bfloat16).

    Returns:
        ``(model, collator, train_cfg)``.
    """
    import hydra  # lazy import: only this helper needs hydra
    from omegaconf import OmegaConf

    train_cfg = OmegaConf.load(str(train_config_path))
    log.info("Loaded saved training config from %s", train_config_path)
    model = hydra.utils.instantiate(train_cfg.policy)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    collator = hydra.utils.instantiate(train_cfg.data_collator, mode=collator_mode)
    return model, collator, train_cfg
