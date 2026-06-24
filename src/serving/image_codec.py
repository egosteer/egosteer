"""JPEG image transport codec for the websocket policy protocol.

Wire format matches EgoSteer-Inference's production client
(websocket_client.py @ commit 8a2f630): each camera's RGB sequence becomes
a self-describing dict with `__image_encoding__: "jpeg_sequence"`, plus a
top-level ``obs["image_compression"]`` metadata key.

Decode is parallelized across frames with a module-level thread pool
(cv2.imdecode releases the GIL). Tune with ``EGOSTEER_JPEG_DECODE_WORKERS``;
set to ``1`` to fall back to single-threaded decode.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

_TAG = "jpeg_sequence"
_DECODE_WORKERS = max(1, int(os.environ.get("EGOSTEER_JPEG_DECODE_WORKERS", "4")))
_DECODE_POOL: ThreadPoolExecutor | None = (
    ThreadPoolExecutor(max_workers=_DECODE_WORKERS, thread_name_prefix="jpeg-decode")
    if _DECODE_WORKERS > 1
    else None
)


def encode_image_field(field: Any, quality: int = 80) -> Any:
    """RGB image (or per-camera dict of them) → JPEG-encoded transport dict.

    Accepts ``(T, H, W, 3)`` or ``(H, W, 3)`` uint8 arrays, or a dict mapping
    camera names to such arrays. Encoder is byte-compatible with the
    production client.
    """
    if isinstance(field, dict):
        return {k: encode_image_field(v, quality) for k, v in field.items()}

    arr = np.asarray(field)
    original_shape = tuple(arr.shape)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 RGB (T,H,W,3) or (H,W,3); got shape={original_shape} dtype={arr.dtype}"
        )

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    frames: list[bytes] = []
    for frame in arr:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        frames.append(encoded.tobytes())
    return {
        "__image_encoding__": _TAG,
        "format": "jpeg",
        "quality": int(quality),
        "shape": original_shape,
        "dtype": "uint8",
        "color_order": "rgb",
        "frames": frames,
        "raw_nbytes": int(arr.nbytes),
        "encoded_nbytes": sum(len(f) for f in frames),
    }


def _decode_one_frame(buf: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode returned None")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_image_field(field: Any) -> Any:
    """Inverse of :func:`encode_image_field`. Pass-through for raw ndarrays."""
    if isinstance(field, np.ndarray):
        return field
    if isinstance(field, dict) and field.get("__image_encoding__") == _TAG:
        frames_bytes = field["frames"]
        if _DECODE_POOL is not None and len(frames_bytes) > 1:
            decoded = list(_DECODE_POOL.map(_decode_one_frame, frames_bytes))
        else:
            decoded = [_decode_one_frame(b) for b in frames_bytes]
        arr = np.stack(decoded, axis=0)
        # Encoder added a leading T dim for 3D inputs; squeeze it back so the
        # round-trip preserves ndim.
        if len(field.get("shape", ())) == 3:
            arr = arr[0]
        return arr
    if isinstance(field, dict):
        return {k: decode_image_field(v) for k, v in field.items()}
    return field


def decode_image_fields_in_obs(obs: dict) -> dict:
    """Decode JPEG-encoded image fields in an observation. No-op for raw clients.

    Detection relies on the protocol-level ``image_compression`` metadata key
    that the JPEG client always sets; raw clients never set it.
    """
    if not isinstance(obs, dict) or "image_compression" not in obs:
        return obs
    out = dict(obs)
    for key in ("image", "depth_image"):
        if key in out:
            out[key] = decode_image_field(out[key])
    out.pop("image_compression", None)
    return out
