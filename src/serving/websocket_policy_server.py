# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py

import asyncio
from contextlib import nullcontext
import http
import logging
import pathlib
import time
import traceback
from typing import Any, Dict

import hydra
import numpy as np
import torch
import torch.nn as nn
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from . import msgpack_numpy
from .image_codec import decode_image_fields_in_obs
from .inference_profiler import InferenceProfiler
from .serving_recorder import ConnectionRecorder, ServingRecorder


logger = logging.getLogger(__name__)


class RuntimeEngine:
    """
    The Runtime Engine.
    It manages the execution context (Device, Autocast) for a Policy.
    """

    def __init__(
        self,
        policy: Any,
        device: torch.device,
        use_autocast: bool,
        warmup_image_shape: tuple[int, int, int],
        warmup_depth_shape: tuple[int, int, int],
        warmup_intrinsic: np.ndarray,
        warmup_camera_setup_mode: str = "single",
        warmup_image_mode: str = "rgb",
    ) -> None:
        self.policy = policy
        self.device = device
        self.use_autocast = use_autocast
        self.warmup_image_shape = tuple(int(x) for x in warmup_image_shape)
        self.warmup_depth_shape = tuple(int(x) for x in warmup_depth_shape)
        self.warmup_intrinsic = np.asarray(warmup_intrinsic, dtype=np.float64)
        self.warmup_camera_setup_mode = str(warmup_camera_setup_mode).lower()
        self.warmup_image_mode = str(warmup_image_mode).lower()
        if self.warmup_camera_setup_mode not in {"single", "both"}:
            raise ValueError(
                "warmup_camera_setup_mode must be 'single' or 'both', "
                f"got {warmup_camera_setup_mode!r}"
            )
        if self.warmup_image_mode not in {"rgb", "rgbd"}:
            raise ValueError(
                "warmup_image_mode must be 'rgb' or 'rgbd', "
                f"got {warmup_image_mode!r}"
            )
        self._profiler: InferenceProfiler | None = None

        self.policy.to(device)

    def _move_to_device(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        return data.to(self.device) if torch.is_tensor(data) else data

    def _autocast_context(self):
        if not self.use_autocast:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def enable_profiling(
        self,
        output_dir: str | pathlib.Path,
        steps: int,
        skip_first: int,
        count_flops: bool = False,
    ) -> None:
        if steps <= 0:
            return
        self._profiler = InferenceProfiler(
            output_dir=output_dir,
            steps=steps,
            skip_first=skip_first,
            device=self.device,
            compile_active=self._compile_is_active(),
            count_flops=count_flops,
        )
        self._profiler.start()

    def _compile_is_active(self) -> bool:
        try:
            return hasattr(self.policy.model, "_orig_mod")
        except Exception:
            return False

    def _step_profiler(self) -> None:
        if self._profiler is None:
            return
        self._profiler.step()
        if self._profiler.done:
            self._profiler = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def _build_dummy_obs(self, instruction: str, rtc: bool) -> Dict[str, Any]:
        shape_meta = self.shape_meta
        rgb_meta = shape_meta["obs"]["rgb"]
        state_meta = shape_meta["obs"]["state"]
        action_meta = shape_meta["action"]

        image = np.zeros((rgb_meta["horizon"], *self.warmup_image_shape), dtype=np.uint8)
        states = np.zeros((state_meta["horizon"], state_meta["shape"][0]), dtype=np.float32)
        prev_action_chunk = np.zeros((action_meta["horizon"], action_meta["shape"][0]), dtype=np.float32)
        intrinsic = self.warmup_intrinsic.copy()

        obs = {
            "image": image,
            "intrinsic": intrinsic,
            "instruction": instruction,
            "states": states,
            "prev_action_chunk": prev_action_chunk if rtc else None,
        }

        if self.warmup_camera_setup_mode == "both":
            # Warmup should exercise the same multimodal path as production dual-camera requests.
            obs["chest_image"] = np.zeros((rgb_meta["horizon"], *self.warmup_image_shape), dtype=np.uint8)
            obs["chest_intrinsic"] = self.warmup_intrinsic.copy()

        depth_meta = shape_meta["obs"].get("depth")
        if depth_meta is not None and self.warmup_image_mode == "rgbd":
            obs["depth"] = np.zeros((depth_meta["horizon"], *self.warmup_depth_shape), dtype=np.uint16)
            if self.warmup_camera_setup_mode == "both":
                obs["chest_depth"] = np.zeros(
                    (depth_meta["horizon"], *self.warmup_depth_shape), dtype=np.uint16
                )

        return obs

    def warmup(self, warmup_iters: int, instruction: str) -> None:
        if warmup_iters <= 0:
            return

        dummy_obs = self._build_dummy_obs(instruction=instruction, rtc=False)
        dummy_obs_rtc = self._build_dummy_obs(instruction=instruction, rtc=True)
        logger.info("Running %d warmup inference iterations", warmup_iters)
        start_time = time.monotonic()
        # first observation does not contain RTC, subsequent ones contain RTC
        for i in range(warmup_iters):
            if i > 0:
                self.infer(dummy_obs_rtc)
            else:
                self.infer(dummy_obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        logger.info("Warmup finished in %.3f ms", (time.monotonic() - start_time) * 1000.0)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """The high-level entry point for inference."""
        inputs = self.prepare_process(obs)
        inputs = self._move_to_device(inputs)

        with self._autocast_context(), torch.inference_mode():
            pred_actions = self.policy(inputs)
        pred_actions = self.post_process(pred_actions.cpu())
        output = {"pred_actions": pred_actions.cpu().float().numpy()[0]}
        attn_grid = self._last_attention_grid
        if attn_grid is not None:
            output["attention_grid"] = attn_grid
        self._step_profiler()
        return output


class EnvWrapper:
    def __init__(
        self,
        policy: Any,
        image_key: str = "image",
        depth_key: str = "depth_image",
        intrinsic_key: str = "camera_intrinsics",
        instruction_key: str = "instruction",
        states_key: str = "states",
        prev_action_chunk_key: str = "action_rtc",
        camera_setup_mode: str = "single",
        image_mode: str = "rgb",
        head_camera_name: str = "head",
        chest_camera_name: str = "chest",
    ) -> None:
        self.policy = policy
        self.image_key = image_key
        self.depth_key = depth_key
        self.intrinsic_key = intrinsic_key
        self.instruction_key = instruction_key
        self.states_key = states_key
        self.prev_action_chunk_key = prev_action_chunk_key
        self.camera_setup_mode = str(camera_setup_mode).lower()
        self.image_mode = str(image_mode).lower()
        self.head_camera_name = head_camera_name
        self.chest_camera_name = chest_camera_name

        if self.camera_setup_mode not in {"single", "both"}:
            raise ValueError(f"camera_setup_mode must be 'single' or 'both', got {camera_setup_mode!r}")
        if self.image_mode not in {"rgb", "rgbd"}:
            raise ValueError(f"image_mode must be 'rgb' or 'rgbd', got {image_mode!r}")

    def _pick_camera_value(self, value: Any, camera_name: str) -> Any:
        if isinstance(value, dict):
            return value.get(camera_name)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self.policy)) | set(super().__dir__()))

    def infer(self, obs: dict) -> dict:
        image_value = obs.get(self.image_key)
        depth_value = obs.get(self.depth_key)
        intrinsic_value = obs.get(self.intrinsic_key)

        head_image = self._pick_camera_value(image_value, self.head_camera_name)
        head_intrinsic = self._pick_camera_value(intrinsic_value, self.head_camera_name)

        mapped_obs = {
            "image": head_image,
            "depth": (
                self._pick_camera_value(depth_value, self.head_camera_name)
                if self.image_mode == "rgbd" else None
            ),
            "intrinsic": head_intrinsic,
            "instruction": obs.get(self.instruction_key),
            "states": obs.get(self.states_key),
            # RTC condition: executed action prefix (None on first step)
            "prev_action_chunk": obs.get(self.prev_action_chunk_key),
        }

        if self.camera_setup_mode == "both":
            chest_image = self._pick_camera_value(image_value, self.chest_camera_name)
            chest_intrinsic = self._pick_camera_value(intrinsic_value, self.chest_camera_name)
            chest_depth = (
                self._pick_camera_value(depth_value, self.chest_camera_name)
                if self.image_mode == "rgbd" else None
            )

            # Canonical dual-view keys at serving boundary.
            mapped_obs["chest_image"] = chest_image
            mapped_obs["chest_intrinsic"] = chest_intrinsic
            mapped_obs["chest_depth"] = chest_depth

        return self.policy.infer(mapped_obs)


def _resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_engine(policy_cfg: Any, serving_cfg: Any) -> Any:
    return RuntimeEngine(
        hydra.utils.instantiate(policy_cfg),
        device=_resolve_device(serving_cfg.device),
        use_autocast=serving_cfg.autocast,
        warmup_image_shape=tuple(serving_cfg.warmup_image_shape),
        warmup_depth_shape=tuple(serving_cfg.warmup_depth_shape),
        warmup_intrinsic=np.asarray(serving_cfg.warmup_intrinsic, dtype=np.float64),
        warmup_camera_setup_mode=getattr(serving_cfg, "warmup_camera_setup_mode", "single"),
        warmup_image_mode=getattr(serving_cfg, "warmup_image_mode", "rgb"),
    )


def create_env_wrapper(policy: Any, wrapper_cfg: Any) -> Any:
    return EnvWrapper(
        policy=policy,
        image_key=wrapper_cfg.image_key,
        depth_key=wrapper_cfg.depth_key,
        intrinsic_key=wrapper_cfg.intrinsic_key,
        instruction_key=wrapper_cfg.instruction_key,
        states_key=wrapper_cfg.states_key,
        prev_action_chunk_key=wrapper_cfg.prev_action_chunk_key,
        camera_setup_mode=getattr(wrapper_cfg, "camera_setup_mode", "single"),
        image_mode=getattr(wrapper_cfg, "image_mode", "rgb"),
        head_camera_name=getattr(wrapper_cfg, "head_camera_name", "head"),
        chest_camera_name=getattr(wrapper_cfg, "chest_camera_name", "chest"),
    )


class WebsocketPolicyServer:
    """Serve a policy with the websocket protocol."""

    def __init__(
        self,
        policy: Any,
        recorder: ServingRecorder | None,
        log_obs_details: bool,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._recorder = recorder
        self._log_obs_details = log_obs_details
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    @staticmethod
    def _format_obs_details(obs: Dict[str, Any]) -> str:
        import numpy as np

        lines = ["\nObservation data details:"]
        lines.append(f"  Total fields: {len(obs)}")
        lines.append(f"  Field list: {list(obs.keys())}")
        lines.append("-" * 80)

        for key, value in obs.items():
            lines.append(f"\nField: '{key}'")

            if value is None:
                lines.append("  Type: None")
                continue

            value_type = type(value).__name__
            lines.append(f"  Python type: {value_type}")

            if hasattr(value, "shape"):
                lines.append(f"  Shape: {value.shape}")

                if hasattr(value, "dtype"):
                    lines.append(f"  Dtype: {value.dtype}")

                if hasattr(value, "nbytes"):
                    size_bytes = value.nbytes
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} bytes"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes / 1024:.2f} KB"
                    else:
                        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                    lines.append(f"  Memory size: {size_str}")

                try:
                    if hasattr(value, "size") and value.size == 0:
                        lines.append("  Status: empty array")
                    elif np.issubdtype(value.dtype, np.number):
                        lines.append("  Numerical statistics:")
                        lines.append(f"    - Min: {float(value.min()):.6f}")
                        lines.append(f"    - Max: {float(value.max()):.6f}")
                        lines.append(f"    - Mean: {float(value.mean()):.6f}")
                        if hasattr(value, "std"):
                            lines.append(f"    - Std: {float(value.std()):.6f}")
                    else:
                        lines.append("  Data type: non-numeric")
                except Exception as exc:
                    lines.append(f"  Statistics: unable to compute ({str(exc)})")

            elif isinstance(value, (list, tuple)):
                lines.append(f"  Length: {len(value)}")
                if len(value) > 0:
                    lines.append(f"  First element type: {type(value[0]).__name__}")

            elif isinstance(value, str):
                lines.append(f"  Length: {len(value)} characters")
                preview = value[:50] + "..." if len(value) > 50 else value
                lines.append(f"  Content preview: {preview}")

            elif isinstance(value, (int, float)):
                lines.append(f"  Value: {value}")

            else:
                lines.append(f"  Description: {str(value)[:100]}")

        lines.append("-" * 80)
        return "\n".join(lines)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        connection_recorder: ConnectionRecorder | None = None

        if self._recorder is not None:
            try:
                connection_recorder = self._recorder.open_connection(websocket.remote_address)
            except Exception:
                logger.exception("Failed to initialize serving recorder for %s", websocket.remote_address)

        await websocket.send(packer.pack(self._metadata))

        prev_send_time = None
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                recv_start = time.monotonic()
                raw_obs = await websocket.recv()
                recv_wait_time = time.monotonic() - recv_start
                obs = msgpack_numpy.unpackb(raw_obs)

                # JPEG -> ndarray. Pass-through when client sent raw images so
                # downstream EnvWrapper / RuntimeEngine never see encoded dicts.
                decode_start = time.monotonic()
                obs = decode_image_fields_in_obs(obs)
                decode_image_ms = (time.monotonic() - decode_start) * 1000.0

                if self._log_obs_details:
                    logger.info("Received observation from %s%s", websocket.remote_address, self._format_obs_details(obs))

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                action["server_timing"] = {
                    "recv_wait_ms": recv_wait_time * 1000,
                    "decode_image_ms": decode_image_ms,
                    "infer_ms": infer_time * 1000,
                }
                if prev_send_time is not None:
                    action["server_timing"]["prev_send_ms"] = prev_send_time * 1000
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # Remove attention grid before packing (not serializable via
                # msgpack and only used for local recording).
                attention_grid = action.pop("attention_grid", None)

                pack_start = time.monotonic()
                packed_action = packer.pack(action)
                pack_time = time.monotonic() - pack_start
                action["server_timing"]["pack_ms"] = pack_time * 1000
                action["server_timing"]["pre_send_total_ms"] = (time.monotonic() - start_time) * 1000

                record_time = 0.0
                if connection_recorder is not None:
                    try:
                        record_start = time.monotonic()
                        connection_recorder.record(obs, action, attention_grid=attention_grid)
                        record_time = time.monotonic() - record_start
                    except Exception:
                        logger.exception("Failed to record request for %s", websocket.remote_address)

                action["server_timing"]["record_ms"] = record_time * 1000

                packed_action = packer.pack(action)
                send_start = time.monotonic()
                await websocket.send(packed_action)
                prev_send_time = time.monotonic() - send_start
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
