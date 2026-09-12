"""Sequential PyAV video windows, including LeRobot's 12-bit depth videos."""

import math
from collections import OrderedDict

import av
import numpy as np


def depth_parameters(feature):
    """Read persisted encoder parameters; never guess them from the codec."""
    info = {**(feature.get("video_info") or {}), **(feature.get("info") or {})}
    if not (info.get("is_depth_map") or info.get("video.is_depth_map")):
        raise ValueError("depth feature must declare is_depth_map=true")
    params = {}
    for key in ("depth_min", "depth_max", "shift", "use_log"):
        if f"video.{key}" not in info:
            raise ValueError(f"depth feature missing persisted video.{key}")
        params[key] = info[f"video.{key}"]
    lo, hi, shift = (float(params[k]) for k in ("depth_min", "depth_max", "shift"))
    if (not all(math.isfinite(v) for v in (lo, hi, shift)) or not 0 <= lo < hi
            or not isinstance(params["use_log"], bool)
            or (params["use_log"] and lo + shift <= 0)):
        raise ValueError(f"invalid depth quantization parameters: {params}")
    return params


def dequantize_depth_m(codes, *, depth_min, depth_max, shift, use_log):
    """Invert the official LeRobot 0.6.1 12-bit mapping, output float32 metres.

    Release specification §5.2 reserves zero as invalid. Restore these pixels
    to zero explicitly (upstream's generic inverse maps code zero to depth_min).
    Valid codes follow the same affine/logarithmic inverse as depth_utils.py.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.dtype != np.uint16 or np.any(codes > 4095):
        raise ValueError("expected uint16 HW depth codes in [0,4095]")
    lo, hi, shift = float(depth_min), float(depth_max), float(shift)
    if use_log:
        offset = math.log(lo + shift)
        scale = (math.log(hi + shift) - offset) / 4095
    else:
        offset, scale = lo, (hi - lo) / 4095
    result = codes.astype(np.float32) * scale + offset
    if use_log:
        np.exp(result, out=result)
        result -= shift
    np.clip(result, lo, hi, out=result)
    result[codes == 0] = 0.0
    return result


class VideoFrameCache:
    """Bounded frame LRU with a persistent sequential decoder per video file."""

    def __init__(self, max_frames=256, max_readers=4):
        if max_frames < 1 or max_readers < 1:
            raise ValueError("video cache sizes must be positive")
        self.max_frames = int(max_frames)
        self.max_readers = int(max_readers)
        self.frames = OrderedDict()
        self.readers = OrderedDict()

    def close(self):
        for state in self.readers.values():
            state["container"].close()
        self.readers.clear()
        self.frames.clear()

    def _get_reader(self, path, fps):
        if path not in self.readers:
            container = av.open(path)
            stream = container.streams.video[0]
            stream.thread_count = 1
            if stream.average_rate is None or not np.isclose(float(stream.average_rate), fps):
                container.close()
                raise ValueError(f"{path}: video fps does not match info.json fps={fps}")
            self.readers[path] = {
                "container": container,
                "stream": stream,
                "decoder": None,
                "last": -1,
            }
            while len(self.readers) > self.max_readers:
                self.readers.popitem(last=False)[1]["container"].close()
        self.readers.move_to_end(path)
        return self.readers[path]

    def _decode_interval(self, path, first, last, fps, depth):
        """Seek when needed, then decode forward while retaining decoder position."""
        state = self._get_reader(path, fps)
        stream = state["stream"]
        needs_seek = state["decoder"] is None or first <= state["last"] or first - state["last"] > fps
        if needs_seek:
            pts = int((first / fps) / float(stream.time_base))
            state["container"].seek(pts, stream=stream, backward=True, any_frame=False)
            state["decoder"] = state["container"].decode(stream)
            state["last"] = -1

        for frame in state["decoder"]:
            if frame.pts is None:
                raise ValueError(f"{path}: missing video PTS")
            position = float(frame.pts * stream.time_base) * fps
            index = round(position)
            if not np.isclose(position, index, atol=1e-3):
                raise ValueError(f"{path}: frame PTS is not aligned to the declared fps")
            state["last"] = index
            if index < first:
                continue
            if index > last:
                break
            if depth:
                if frame.format.name != "gray12le":
                    raise ValueError(f"{path}: expected gray12le depth, got {frame.format.name}")
                array = dequantize_depth_m(frame.to_ndarray(format="gray12le"), **depth)
            else:
                array = frame.to_ndarray(format="rgb24")
            yield index, array
            if index == last:
                break

    def read(self, path, indices, fps, depth=None):
        path = str(path)
        indices = [int(i) for i in indices]
        if not indices or min(indices) < 0:
            raise ValueError("video indices must be nonempty and nonnegative")
        kind = tuple(sorted(depth.items())) if depth else None
        need = sorted(set(indices))
        key = lambda i: (path, kind, i)
        resolved = {i: self.frames[key(i)] for i in need if key(i) in self.frames}
        for i in resolved:
            self.frames.move_to_end(key(i))
        missing = [i for i in need if i not in resolved]
        if missing:
            requested = set(missing)
            for i, array in self._decode_interval(path, missing[0], missing[-1], fps, depth):
                self.frames[key(i)] = array
                if i in requested:
                    resolved[i] = array
                while len(self.frames) > self.max_frames:
                    self.frames.popitem(last=False)
            if set(need) - resolved.keys():
                raise ValueError(f"{path}: missing video frames {sorted(set(need) - resolved.keys())}")
        return np.stack([resolved[i] for i in indices])
