from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
import re
from typing import Any

import cv2
import numpy as np

from src.utils.visual_attention import overlay_attention


logger = logging.getLogger(__name__)


class ServingRecorder:
    def __init__(
        self,
        root_dir: str | pathlib.Path,
        image_key: str = "image",
        depth_key: str = "depth_image",
    ) -> None:
        self._root_dir = pathlib.Path(root_dir).expanduser()
        self._image_key = image_key
        self._depth_key = depth_key
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def open_connection(self, remote_address: Any) -> "ConnectionRecorder":
        timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        remote_label = self._format_remote_address(remote_address)
        connection_dir = self._root_dir / f"conn_{timestamp}_{remote_label}"
        connection_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Recording connection data to %s", connection_dir)
        return ConnectionRecorder(
            connection_dir,
            remote_address=remote_address,
            image_key=self._image_key,
            depth_key=self._depth_key,
        )

    @staticmethod
    def _format_remote_address(remote_address: Any) -> str:
        if isinstance(remote_address, tuple):
            parts = [str(part) for part in remote_address if part not in (None, "")]
            raw = "_".join(parts)
        else:
            raw = str(remote_address or "unknown")
        return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_") or "unknown"


class ConnectionRecorder:
    def __init__(
        self,
        connection_dir: pathlib.Path,
        remote_address: Any,
        image_key: str,
        depth_key: str,
    ) -> None:
        self._connection_dir = connection_dir
        self._remote_address = remote_address
        self._image_key = image_key
        self._depth_key = depth_key
        self._request_index = 0

    def record(
        self,
        obs: dict[str, Any],
        response: dict[str, Any],
        attention_grid: np.ndarray | None = None,
    ) -> pathlib.Path:
        self._request_index += 1
        request_dir = self._connection_dir / f"req_{self._request_index:06d}"
        request_dir.mkdir(parents=True, exist_ok=True)

        request_payload = {
            "request_index": self._request_index,
            "recorded_at": self._timestamp(),
            "remote_address": self._json_remote_address(),
            "obs_summary": {key: self._summarize_value(value) for key, value in obs.items()},
        }
        request_payload.update(self._build_request_fields(obs))

        preview_written = self._save_observation_image(obs, request_dir / "observation.jpg")
        if preview_written:
            request_payload["observation_image"] = "observation.jpg"

        response_payload = {
            "request_index": self._request_index,
            "recorded_at": self._timestamp(),
        }
        response_payload.update(self._to_jsonable(response))

        if attention_grid is not None:
            views = self._last_frame_rgb_per_view(obs)
            if views:
                overlay = self._build_attention_overlay(views, attention_grid)
                cv2.imwrite(
                    str(request_dir / "attention_overlay.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
                response_payload["attention_overlay"] = "attention_overlay.jpg"

        self._write_json(request_dir / "request.json", request_payload)
        self._write_json(request_dir / "response.json", response_payload)
        return request_dir

    def _last_frame_rgb(self, obs: dict[str, Any]) -> np.ndarray | None:
        """Extract the last RGB frame from obs as uint8 [H, W, 3]."""
        image = self._extract_camera_view(obs.get(self._image_key), "head")
        if image is None:
            return None
        frame = np.asarray(image, dtype=np.float32)[-1]
        return self._normalize_rgb(frame)

    def _last_frame_rgb_per_view(self, obs: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
        """Return [(camera_name, last_frame_uint8_rgb), ...] in head-first order."""
        image = obs.get(self._image_key)
        if image is None:
            return []
        if isinstance(image, dict):
            views = []
            for cam in ("head", "chest"):
                v = image.get(cam)
                if v is not None:
                    views.append((cam, self._normalize_rgb(np.asarray(v, dtype=np.float32)[-1])))
            return views
        return [("head", self._normalize_rgb(np.asarray(image, dtype=np.float32)[-1]))]

    def _build_attention_overlay(
        self, views: list[tuple[str, np.ndarray]], attention_grid: np.ndarray,
    ) -> np.ndarray:
        """Overlay each view's last attention frame on its last RGB frame, stacked
        in a 2×2 grid:

          top row:    per-view independent normalization (existing behavior)
          bottom row: joint normalization across all views

        attention_grid is [T_total, tH, tW] with per-view T equal; views are head-first.
        """
        n = len(views)
        T_total = int(attention_grid.shape[0])
        if n == 0 or T_total % n != 0:
            return overlay_attention(views[0][1], attention_grid[-1])
        T_per_view = T_total // n

        # Top row: per-view independent normalization
        top_overlays = [
            overlay_attention(frame, attention_grid[(i + 1) * T_per_view - 1])
            for i, (_, frame) in enumerate(views)
        ]
        top_row = np.concatenate(top_overlays, axis=1)

        # Bottom row: joint normalization across all views
        # Upsample all attention maps and compute shared percentiles
        attn_up_parts = []
        for i, (_, frame) in enumerate(views):
            attn = attention_grid[(i + 1) * T_per_view - 1]
            attn_np = np.asarray(attn, dtype=np.float32)
            H, W = frame.shape[:2]
            attn_up_parts.append(cv2.resize(attn_np, (W, H), interpolation=cv2.INTER_LINEAR))
        all_attn = np.concatenate([a.ravel() for a in attn_up_parts])
        lo, hi = np.percentile(all_attn, (2.0, 98.0))

        bottom_overlays = [
            overlay_attention(
                frame, attention_grid[(i + 1) * T_per_view - 1], vmin=lo, vmax=hi,
            )
            for i, (_, frame) in enumerate(views)
        ]
        bottom_row = np.concatenate(bottom_overlays, axis=1)

        return np.concatenate([top_row, bottom_row], axis=0)

    @staticmethod
    def _extract_camera_view(value: Any, camera_name: str) -> Any:
        if isinstance(value, dict):
            if camera_name in value:
                return value[camera_name]
            # Fallback for non-standard payloads.
            if value:
                return next(iter(value.values()))
            return None
        return value

    def _build_request_fields(self, obs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in obs.items():
            if key in {self._image_key, self._depth_key}:
                continue
            payload[key] = self._to_jsonable(value)
        return payload

    def _save_observation_image(self, obs: dict[str, Any], output_path: pathlib.Path) -> bool:
        image = obs.get(self._image_key)
        if image is None:
            return False

        if isinstance(image, dict):
            head_canvas = self._prepare_rgb_canvas(self._extract_camera_view(image, "head"))
            chest_view = self._extract_camera_view(image, "chest")
            if chest_view is not None:
                chest_canvas = self._prepare_rgb_canvas(chest_view)
                image_canvas = np.concatenate([head_canvas, chest_canvas], axis=1)
            else:
                image_canvas = head_canvas

            depth = obs.get(self._depth_key)
            if isinstance(depth, dict):
                head_depth = self._extract_camera_view(depth, "head")
                chest_depth = self._extract_camera_view(depth, "chest")
                depth_parts = []
                if head_depth is not None:
                    depth_parts.append(self._prepare_depth_canvas(head_depth))
                if chest_depth is not None:
                    depth_parts.append(self._prepare_depth_canvas(chest_depth))
                if depth_parts:
                    depth_canvas = np.concatenate(depth_parts, axis=1)
                    canvas = np.concatenate([image_canvas, depth_canvas], axis=1)
                else:
                    canvas = image_canvas
            else:
                canvas = image_canvas
        else:
            image_canvas = self._prepare_rgb_canvas(image)
            depth = obs.get(self._depth_key)
            if depth is not None:
                depth_canvas = self._prepare_depth_canvas(depth)
                canvas = np.concatenate([image_canvas, depth_canvas], axis=1)
            else:
                canvas = image_canvas

        return bool(cv2.imwrite(str(output_path), canvas))

    @staticmethod
    def _prepare_rgb_canvas(value: Any) -> np.ndarray:
        # Expected image shape: [T, H, W, 3], with values in [0, 1] or [0, 255].
        array = np.asarray(value, dtype=np.float32)
        assert array.ndim == 4 and array.shape[-1] == 3, (
            f"Expected image shape [T, H, W, 3], got {array.shape}"
        )
        image = array[-1]
        image = ConnectionRecorder._normalize_rgb(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _prepare_depth_canvas(value: Any) -> np.ndarray:
        # Expected depth_image shape: [T, H, W, 1]. Save it as a per-frame min/max normalized preview.
        array = np.asarray(value, dtype=np.float32)
        assert array.ndim == 4 and array.shape[-1] == 1, (
            f"Expected depth_image shape [T, H, W, 1], got {array.shape}"
        )
        depth = array[-1, :, :, 0]
        depth_uint8 = ConnectionRecorder._normalize_grayscale(depth)
        return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _normalize_rgb(array: np.ndarray) -> np.ndarray:
        data = np.asarray(array, dtype=np.float32)
        if data.max() <= 1.0:
            data = data * 255.0
        return np.clip(data, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _normalize_grayscale(array: np.ndarray) -> np.ndarray:
        data = np.asarray(array, dtype=np.float32)
        min_value = float(data.min())
        max_value = float(data.max())
        if max_value <= min_value:
            return np.zeros(data.shape, dtype=np.uint8)
        scaled = (data - min_value) / (max_value - min_value)
        return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): ConnectionRecorder._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ConnectionRecorder._to_jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _summarize_value(value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "None"}

        summary: dict[str, Any] = {"type": type(value).__name__}
        if isinstance(value, np.ndarray):
            summary["shape"] = list(value.shape)
            summary["dtype"] = str(value.dtype)
            summary["size"] = int(value.size)
            if value.size > 0 and np.issubdtype(value.dtype, np.number):
                finite_mask = np.isfinite(value)
                if finite_mask.any():
                    finite = value[finite_mask]
                    summary["min"] = float(np.min(finite))
                    summary["max"] = float(np.max(finite))
            return summary
        if isinstance(value, (list, tuple)):
            summary["length"] = len(value)
            return summary
        if isinstance(value, str):
            summary["length"] = len(value)
            return summary
        if isinstance(value, np.generic):
            summary["dtype"] = str(value.dtype)
            summary["value"] = value.item()
            return summary
        if isinstance(value, (int, float, bool)):
            summary["value"] = value
            return summary
        return summary

    @staticmethod
    def _timestamp() -> str:
        return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _json_remote_address(self) -> Any:
        if isinstance(self._remote_address, tuple):
            return [self._to_jsonable(part) for part in self._remote_address]
        return self._to_jsonable(self._remote_address)
