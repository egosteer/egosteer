'''
Normalizer utility functions for EgoSteer datasets.
'''

import itertools

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.common.normalizer import LinearNormalizer


def get_normalizer(dataloader_cfg, normalizer_dataset, return_metadata=False):
    assert normalizer_dataset is not None, "normalizer_dataset must be provided"

    dataloader_cfg = dict(dataloader_cfg)
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    dataloader_cfg.setdefault("pin_memory", False)
    if num_workers > 0:
        dataloader_cfg.setdefault("persistent_workers", True)
        dataloader_cfg.setdefault("prefetch_factor", 4)

    dataloader = DataLoader(
        normalizer_dataset,
        collate_fn=normalizer_dataset.get_collator(),
        **dataloader_cfg,
    )
    dataloader_iter = iter(dataloader)
    try:
        first_batch = next(dataloader_iter)
    except StopIteration as exc:
        raise ValueError("No data to calculate normalizer") from exc

    normalizer = LinearNormalizer()
    normalizer_keys = [
        key for key, value in first_batch.items()
        if not key.startswith("_") and isinstance(value, (torch.Tensor, np.ndarray))
    ]
    if not normalizer_keys:
        raise ValueError("No tensor-like batch entries found for normalizer fitting")

    metadata = {
        "normalizer_keys": list(normalizer_keys),
        "current_frames_scanned": 0,
        "effective_rows": {key: 0 for key in normalizer_keys},
    }

    normalizer.start_streaming_fit(keys=normalizer_keys)
    for batch in tqdm(
        itertools.chain([first_batch], dataloader_iter),
        desc="Calculating normalizer",
    ):
        input_data = {
            key: value.reshape(-1, value.shape[-1])
            for key, value in batch.items()
            if key in normalizer_keys and isinstance(value, (torch.Tensor, np.ndarray))
        }
        batch_num_samples = batch.get("_batch_num_samples")
        if batch_num_samples is None:
            batch_num_samples = batch[normalizer_keys[0]].shape[0]
        metadata["current_frames_scanned"] += int(batch_num_samples)
        for key, value in input_data.items():
            metadata["effective_rows"][key] += int(value.shape[0])
        normalizer.update_streaming_fit(input_data)
    normalizer.finish_streaming_fit()

    for key in normalizer_keys:
        if key in ["states", "actions", "motions"]:
            normalizer.ignore_dim(key=key, dim=slice(6, 18))
        else:
            raise ValueError(f"Unsupported key: {key}")

    if return_metadata:
        return normalizer, metadata
    return normalizer
