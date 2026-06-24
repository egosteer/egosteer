from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def compile_module_list(module_list: nn.ModuleList, compile_kwargs: Mapping[str, Any]) -> None:
    if not isinstance(module_list, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList, got {type(module_list).__name__}.")
    for idx, block in enumerate(module_list):
        module_list[idx] = torch.compile(block, **compile_kwargs)
