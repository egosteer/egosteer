"""WebSocket smoke-test client.

Sends randomized observations (480x640 dual-camera by default) and prints
server inference latency. Wire format matches the production client; pass
``--image-format jpeg`` to test the compressed transport path.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any, Dict, Tuple

import numpy as np
import websockets.asyncio.client as _client

from . import msgpack_numpy
from .image_codec import encode_image_field


# Model-side constants (EgoSteer inference shape_meta). Override the source
# here if the deployed model changes; not exposed as CLI flags to keep the
# smoke-test client minimal.
STATE_DIM = 48
ACTION_DIM = 48
ACTION_HORIZON = 32

# Typical 480x640 RealSense intrinsics (fx=fy=388, cx=320, cy=240). The server
# accepts any 3x3 or flat (fx, fy, cx, cy); this constant is just a placeholder.
DEFAULT_INTRINSICS = np.array(
    [[388.0, 0.0, 320.0], [0.0, 388.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _parse_shape(value: str) -> Tuple[int, ...]:
    shape = tuple(int(p) for p in value.split(",") if p.strip())
    if not shape or any(d <= 0 for d in shape):
        raise argparse.ArgumentTypeError(f"invalid shape: {value!r}")
    return shape


def _per_camera(value_factory):
    """Return either a per-camera dict or a single value, depending on args."""
    return {"head": value_factory(), "chest": value_factory()}


def _make_obs(image_shape, camera_setup, image_mode, state_horizon, rtc_delay, instruction):
    rng_rgb = lambda: np.random.randint(0, 256, size=image_shape, dtype=np.uint8)
    rng_depth = lambda: np.random.randint(300, 1500,
                                          size=(*image_shape[:-1], 1), dtype=np.uint16)

    obs: Dict[str, Any] = {
        "instruction": instruction,
        "states": np.random.normal(0, 0.35, (state_horizon, STATE_DIM)).clip(-1, 1).astype(np.float32),
        "action_rtc": np.zeros((rtc_delay, ACTION_DIM), dtype=np.float32) if rtc_delay > 0 else None,
    }
    if camera_setup == "both":
        obs["image"] = _per_camera(rng_rgb)
        obs["camera_intrinsics"] = _per_camera(DEFAULT_INTRINSICS.copy)
        if image_mode == "rgbd":
            obs["depth_image"] = _per_camera(rng_depth)
    else:
        obs["image"] = rng_rgb()
        obs["camera_intrinsics"] = DEFAULT_INTRINSICS.copy()
        if image_mode == "rgbd":
            obs["depth_image"] = rng_depth()
    return obs


def _maybe_compress(obs, image_format, jpeg_quality):
    if image_format not in ("raw", "jpeg"):
        raise ValueError(f"unknown --image-format {image_format!r}")
    if image_format != "jpeg" or "image" not in obs:
        return obs
    out = dict(obs)
    out["image"] = encode_image_field(obs["image"], quality=jpeg_quality)
    out["image_compression"] = {
        "format": "jpeg", "quality": int(jpeg_quality),
        "color_order": "rgb", "field": "image",
    }
    return out


async def _run_client(args: argparse.Namespace) -> None:
    image_shape = _parse_shape(args.image_shape)
    packer = msgpack_numpy.Packer()
    fmt_label = f"jpeg(q={args.jpeg_quality})" if args.image_format == "jpeg" else "raw"
    print(f"[client] camera={args.camera_setup} image={fmt_label} shape={image_shape} "
          f"mode={args.image_mode} state_horizon={args.state_horizon} rtc_delay={args.rtc_delay}")

    async with _client.connect(f"ws://{args.host}:{args.port}", max_size=None, compression=None) as ws:
        print("server metadata:", msgpack_numpy.unpackb(await ws.recv()))
        for idx in range(args.num_requests):
            obs = _make_obs(image_shape, args.camera_setup, args.image_mode,
                            args.state_horizon, args.rtc_delay, args.instruction)
            packed = packer.pack(_maybe_compress(obs, args.image_format, args.jpeg_quality))

            t0 = time.monotonic()
            await ws.send(packed)
            response = msgpack_numpy.unpackb(await ws.recv())
            rtt_ms = (time.monotonic() - t0) * 1000.0
            infer_ms = response.get("server_timing", {}).get("infer_ms")
            infer_str = f"{infer_ms:.1f}ms" if infer_ms is not None else "?"
            print(f"[{idx}] send={len(packed)/1024:.0f}KB infer={infer_str} rtt={rtt_ms:.1f}ms")

            if args.sleep_ms > 0:
                await asyncio.sleep(args.sleep_ms / 1000.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--num-requests", type=int, default=5)
    p.add_argument("--sleep-ms", type=int, default=200)
    p.add_argument("--camera-setup", choices=["single", "both"], default="both")
    p.add_argument("--image-mode", choices=["rgb", "rgbd"], default="rgb")
    p.add_argument("--image-shape", default="6,480,640,3",
                   help="(T, H, W, 3) per camera. T=horizon (server pads if smaller).")
    p.add_argument("--state-horizon", type=int, default=6)
    p.add_argument("--rtc-delay", type=int, default=0,
                   help=f"action_rtc length (1..{ACTION_HORIZON}); 0 disables RTC.")
    p.add_argument("--image-format", choices=["raw", "jpeg"], default="raw",
                   help="jpeg: per-frame JPEG encode + dict wire format "
                        "(compatible with EgoSteer-Inference production client).")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--instruction", default="grasp the yellow toy")
    return p.parse_args()


def main() -> None:
    asyncio.run(_run_client(_parse_args()))


if __name__ == "__main__":
    main()
