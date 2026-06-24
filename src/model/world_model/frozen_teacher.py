"""Frozen visual teacher for world model supervision.

Wraps a pretrained DINOv3 ViT as a frozen feature extractor. All parameters
are frozen at init; forward runs under ``torch.no_grad``.  GPU-side
preprocessing converts uint8 BHWC images to ImageNet-normalized BCHW floats.

# Source: https://huggingface.co/docs/transformers/en/model_doc/dinov3
# DINOv3 uses RoPE (native multi-resolution), patch_size=16, 4 register tokens.
# Output layout: last_hidden_state = [CLS, reg0..reg3, patch_0..patch_N]
# Spatial tokens = last_hidden_state[:, 1 + num_register_tokens:, :]
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FrozenVisualTeacher(nn.Module):
    """Frozen DINOv3 ViT as visual representation target.

    Args:
        model_name_or_path: HF model ID or local path for DINOv3 ViT.
        feature_dim: Expected output feature dimension (for assertions).
        torch_dtype: Dtype string for model weights (e.g. "bfloat16").
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        feature_dim: int = 1024,
        torch_dtype: str = "bfloat16",
        target_size: tuple[int, int] | list[int] | None = None,
    ):
        super().__init__()
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        try:
            from transformers import AutoModel, AutoConfig
        except ImportError as exc:
            raise ImportError(
                "FrozenVisualTeacher requires transformers >= 4.56.0 with DINOv3 support."
            ) from exc

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.patch_size = config.patch_size
        self.num_register_tokens = getattr(config, "num_register_tokens", 0)
        self.feature_dim = config.hidden_size
        assert self.feature_dim == feature_dim, (
            f"Model hidden_size={self.feature_dim} != expected feature_dim={feature_dim}"
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            dtype=model_dtype,
        )
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Target size for resizing input images (ensures patch alignment)
        self.target_size = tuple(target_size) if target_size is not None else None
        if self.target_size is not None:
            tH, tW = self.target_size
            assert tH % self.patch_size == 0 and tW % self.patch_size == 0, (
                f"target_size ({tH}, {tW}) must be divisible by patch_size={self.patch_size}"
            )

        # ImageNet normalization constants (not registered as buffers to avoid
        # DTensor conversion issues with FSDP2; created on-the-fly in preprocess).
        self._pixel_mean = (0.485, 0.456, 0.406)
        self._pixel_std = (0.229, 0.224, 0.225)

    def train(self, mode: bool = True) -> "FrozenVisualTeacher":
        # Always keep in eval mode
        return super().train(False)

    def preprocess(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """GPU-side preprocessing: uint8 [B, H, W, 3] -> float [B, 3, H', W'] normalized.

        When target_size is set, images are resized to that resolution (must be
        patch-aligned). Otherwise images are passed through at their original size.
        """
        # BHWC -> BCHW, float [0, 1]
        x = images_uint8.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        # Resize to target_size if specified
        if self.target_size is not None and (x.shape[2], x.shape[3]) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bicubic", align_corners=False)
        # ImageNet normalize (constants created on device to avoid DTensor issues)
        mean = torch.tensor(self._pixel_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self._pixel_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x

    @torch.no_grad()
    def forward(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch features from future frames.

        Args:
            images_uint8: [B, N_fut, H, W, 3] uint8 tensor.

        Returns:
            features: [B, N_fut, N_spatial, feature_dim] float tensor.
                N_spatial = (H // patch_size) * (W // patch_size).
        """
        B, N = images_uint8.shape[:2]

        flat = images_uint8.reshape(B * N, *images_uint8.shape[2:])
        x = self.preprocess(flat)
        # Cast to model dtype for forward pass
        x = x.to(dtype=next(self.model.parameters()).dtype)
        out = self.model(pixel_values=x)
        # Skip CLS token (idx 0) and register tokens (idx 1..num_register_tokens)
        spatial = out.last_hidden_state[:, 1 + self.num_register_tokens:, :]
        return spatial.reshape(B, N, *spatial.shape[1:])
