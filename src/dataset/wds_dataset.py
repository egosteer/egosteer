"""
WebDataset pipeline utilities for EgoSteer training.

Provides sliding window compose for action chunk assembly and
pipeline builders for single/blended WebDataset sources.
"""

import collections
import glob
import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import webdataset as wds
from webdataset.tariterators import base_plus_ext

from .sanity_checks import attach_sample_ctx, build_sample_context


# lowdim.npy layout: base 96D (wrist/hand state+action) + 20D per camera
# (extrinsic 16 + intrinsic 4) appended in meta["cameras"] order.
# cameras[0] is always "head"; legacy shards with no cameras field are head-only.
BASE_LOWDIM_SLICES = {
    'wrist_state':  (0, 18),
    'hand_state':   (18, 48),
    'wrist_action': (48, 66),
    'hand_action':  (66, 96),
}
BASE_LOWDIM_LEN = 96
CAMERA_BLOCK_SIZE = 20


def build_lowdim_slices(cameras):
    """Return {field: (start, end)} for the given camera declaration.
    cameras[0] must be 'head'.
    """
    if not cameras or cameras[0] != "head":
        raise ValueError(f"cameras[0] must be 'head', got {cameras!r}")
    slices = dict(BASE_LOWDIM_SLICES)
    offset = BASE_LOWDIM_LEN
    for cam in cameras:
        slices[f'{cam}_extrinsic'] = (offset, offset + 16)
        slices[f'{cam}_intrinsic'] = (offset + 16, offset + 20)
        offset += CAMERA_BLOCK_SIZE
    return slices


# Head-only default with unprefixed "extrinsic"/"intrinsic" aliases so
# pre-multi-camera audit scripts keep working.
def build_legacy_lowdim_slices():
    slices = build_lowdim_slices(["head"])
    slices["extrinsic"] = slices["head_extrinsic"]
    slices["intrinsic"] = slices["head_intrinsic"]
    return slices


LOWDIM_SLICES = build_legacy_lowdim_slices()


def _is_shard_sequence(shard_patterns):
    """Return True when *shard_patterns* is a non-string sequence of shard entries."""
    return isinstance(shard_patterns, Sequence) and not isinstance(shard_patterns, (str, bytes, os.PathLike))


def _is_glob_pattern(shard_entry):
    """Return True when a shard entry uses shell-style glob wildcards."""
    return any(token in shard_entry for token in "*?[")


def expand_shard_patterns(shard_patterns):
    """Expand shard patterns into explicit shard paths.

    Supports a single string/path-like shard pattern or a sequence of patterns.
    Sequence entries are expanded independently and concatenated so one logical
    dataset subset can keep a single downstream shuffle buffer.

    Glob patterns that match nothing are ignored; literal paths and WebDataset
    brace patterns are preserved as-is.
    """
    if isinstance(shard_patterns, (str, bytes, os.PathLike)):
        shard_entries = [os.fspath(shard_patterns)]
        shard_patterns_metadata = shard_entries[0]
    elif _is_shard_sequence(shard_patterns):
        shard_entries = [os.fspath(entry) for entry in shard_patterns]
        shard_patterns_metadata = list(shard_entries)
    else:
        raise TypeError(
            "shard_patterns must be a string/path-like value or a sequence of shard patterns, "
            f"got {type(shard_patterns)!r}"
        )

    shard_urls = []
    for entry in shard_entries:
        matches = sorted(glob.glob(entry))
        if matches:
            shard_urls.extend(matches)
        elif not _is_glob_pattern(entry):
            shard_urls.append(entry)

    return shard_urls, shard_patterns_metadata


@dataclass
class WindowConfig:
    """Sampling window parameters for sliding window compose."""
    action_horizon: int = 32
    action_stride: int = 1
    state_horizon: int = 16
    state_stride: int = 2
    image_horizon: int = 1
    image_stride: int = 30
    history_pad_mode: str = "repeat"
    action_pad_mode: str = "truncate"
    future_frame_horizon: int = 0
    future_frame_stride: int = 30
    future_frame_pad_mode: str = "repeat"
    # DAgger: drop hq=0 anchor windows; suffix-truncate action/WM at first hq=0.
    # Missing key defaults to 1 (no-op on legacy data).
    dagger_quality_filter: bool = True

    def __post_init__(self):
        valid_modes = {"repeat", "truncate"}
        for name in ("history_pad_mode", "action_pad_mode", "future_frame_pad_mode"):
            value = getattr(self, name)
            if value not in valid_modes:
                raise ValueError(f"Invalid {name}: {value}")

    @property
    def past_size(self):
        """Max number of past frames needed for state and image."""
        return max(
            (self.state_horizon - 1) * self.state_stride,
            (self.image_horizon - 1) * self.image_stride,
        )
    
    @property
    def future_size(self):
        """Number of future frames (including current) needed for action + future frame prediction."""
        action_max = (self.action_horizon - 1) * self.action_stride
        ff_max = self.future_frame_horizon * self.future_frame_stride
        return max(action_max, ff_max) + 1


def decode_sample_fields(sample):
    """Eagerly decode meta.json and lowdim.npy; leave image/depth as bytes
    for post-shuffle decoding."""
    import io as _io
    meta = sample.get("meta.json")
    if isinstance(meta, bytes):
        sample["meta.json"] = json.loads(meta.decode("utf-8"))
    ld = sample.get("lowdim.npy")
    if isinstance(ld, bytes):
        sample["lowdim.npy"] = np.load(_io.BytesIO(ld))
    return sample


def decode_media_fields(sample):
    """Decode image/depth bytes into PIL Image / numpy array.

    Called after the shuffle buffer so that downstream consumers (VLA /
    VLM datasets) always receive decoded media, keeping the deferred-
    decode logic internal to the pipeline.
    """
    from PIL import Image
    import io as _io
    for key in list(sample.keys()):
        val = sample[key]
        if not isinstance(val, bytes):
            continue
        if key.endswith(".jpg") or key.endswith(".jpeg") or key.endswith(".png"):
            sample[key] = Image.open(_io.BytesIO(val)).convert("RGB")
        elif key.endswith(".npy"):
            sample[key] = np.load(_io.BytesIO(val))
    return sample


def gather_history_frames(past, buf, horizon, stride, pad_mode):
    """Gather history frames from past buffer in causal order.

    Args:
        past: deque of past frames (already yielded)
        buf: deque of current + future frames, buf[0] is current
        horizon: number of frames to gather (including current)
        stride: temporal stride between frames
        pad_mode: "repeat" or "truncate" for history features

    Returns:
        List of frames in causal order (oldest first, current last).
        Length is horizon in repeat mode, or <= horizon in truncate mode.
    """
    frames = []
    # Gather past frames: from oldest to newest
    for i in range(horizon - 1, 0, -1):
        offset = i * stride
        if offset <= len(past):
            frames.append(past[-offset])
        elif pad_mode == "repeat":
            # Not enough past: repeat earliest available
            if past:
                frames.append(past[0])
            else:
                frames.append(buf[0])
        # truncate mode: skip missing frames

    # Append current frame at the end
    frames.append(buf[0])
    return frames


def gather_future_refs(buf, horizon, stride, pad_mode, offset_base=0,
                       *, quality_truncate=False):
    """Gather future frame references from the sliding window buffer.

    Unified helper for both action-chunk and future-frame gathering.
    Action chunks use offset_base=0 (buf[0] is the first target),
    future frames use offset_base=stride (skip current frame).

    Pad-mode semantics (`valid_count` follows downstream supervision):
    - ``repeat``: refs is always length ``horizon``; padded positions hold
      copies of the last real frame and are treated as VALID supervision.
      ``valid_count == horizon`` whenever the buffer is non-empty.
    - ``truncate``: refs has only the truly-available refs; ``valid_count``
      equals the real number of in-bound offsets. Downstream masks skip
      the invalid tail.

    ``quality_truncate`` (DAgger): break at first high_quality=0 (missing key
    defaults to 1). Hard cut; episode-tail shortness still uses pad_mode.

    Returns:
        (refs, valid_count): gathered references and count of positions
        treated as valid supervision.
    """
    refs = []
    valid_count = 0
    for i in range(horizon):
        offset = offset_base + i * stride
        if offset < len(buf):
            cand = buf[offset]
            if quality_truncate and int(cand["meta.json"].get("high_quality", 1)) == 0:
                break
            refs.append(cand)
            valid_count += 1
        elif pad_mode == "repeat":
            refs.append(buf[-1])
            valid_count += 1
    return refs, valid_count


def build_sample_from_window(buf, past, config):
    """Build a training sample from the sliding window buffer.

    RGB/depth stay as frame refs until ``materialize_sample_media`` runs
    post-shuffle. chest_* calibration keys appear iff meta["cameras"]
    declares chest; media decode is driven by key presence.
    """
    current = buf[0]
    meta = current["meta.json"]
    cameras = meta.get("cameras", ["head"])
    lowdim_slices = build_lowdim_slices(cameras)

    # Action chunk: len(lowdims_full) is the valid count.
    action_refs, _ = gather_future_refs(
        buf, config.action_horizon, config.action_stride, config.action_pad_mode,
        offset_base=0,
        quality_truncate=config.dagger_quality_filter,
    )
    lowdims_full = np.stack([frame["lowdim.npy"] for frame in action_refs], axis=0)

    state_frames = gather_history_frames(
        past, buf, config.state_horizon, config.state_stride, config.history_pad_mode)
    state_lds = np.stack([f["lowdim.npy"] for f in state_frames], axis=0)

    image_frames = gather_history_frames(
        past, buf, config.image_horizon, config.image_stride, config.history_pad_mode)
    image_frame_refs = tuple(image_frames)

    future_frame_refs = None
    future_lowdims = None
    if config.future_frame_horizon > 0:
        ff_refs, _ = gather_future_refs(
            buf, config.future_frame_horizon, config.future_frame_stride,
            config.future_frame_pad_mode, offset_base=config.future_frame_stride,
            quality_truncate=config.dagger_quality_filter,
        )
        if ff_refs:
            future_frame_refs = tuple(ff_refs)
            future_lowdims = np.stack(
                [frame["lowdim.npy"] for frame in ff_refs], axis=0
            )

    # head_* map to canonical extrinsic/intrinsic; chest_* pass through.
    ld = current["lowdim.npy"]
    result = {}
    for field, (s, e) in lowdim_slices.items():
        if field.endswith("_state"):
            result[field] = state_lds[:, s:e].astype(np.float32)
        elif field.endswith("_action"):
            result[field] = lowdims_full[:, s:e].astype(np.float32)
        elif field == "head_extrinsic":
            result["extrinsic"] = ld[s:e].astype(np.float32)
        elif field == "head_intrinsic":
            result["intrinsic"] = ld[s:e].astype(np.float32)
        elif field in ("chest_extrinsic", "chest_intrinsic"):
            result[field] = ld[s:e].astype(np.float32)

    result.update({
        "instruction": meta["instruction"],
        "instruction_num": meta["instruction_num"],
        # 1=left, 2=right, 3=both; human datasets only.
        "presence": int(meta.get("presence", 3)),
        "dataset_name": meta.get("dataset_name", ""),
        "episode_index": meta.get("episode_index", 0),
        # Propagate webdataset locators so downstream skip logs can pinpoint
        # the exact tar shard + current-frame key for offline triage.
        "shard_url": current.get("__url__", ""),
        "__key__": current.get("__key__", ""),
    })

    result["image_frame_refs"] = image_frame_refs
    if future_frame_refs is not None:
        result["future_frame_refs"] = future_frame_refs
        # Per-camera future extrinsics [K_raw, 16] for WM motion conditioning.
        if future_lowdims is not None:
            hs, he = lowdim_slices["head_extrinsic"]
            result["future_head_extrinsic"] = future_lowdims[:, hs:he].astype(np.float32)
            if "chest_extrinsic" in lowdim_slices:
                bs, be = lowdim_slices["chest_extrinsic"]
                result["future_chest_extrinsic"] = future_lowdims[:, bs:be].astype(np.float32)
    return result


def decode_image_bytes(raw):
    """Decode a single image from raw JPEG bytes (cv2.imdecode, ~2x faster
    than PIL) or pass through an already-decoded array."""
    if isinstance(raw, bytes):
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode failed on image bytes")
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    else:
        image = np.array(raw, copy=True)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def decode_depth_bytes(raw):
    """Decode a single depth map from raw npy bytes or numpy array."""
    if isinstance(raw, bytes):
        import io
        return np.load(io.BytesIO(raw))
    return np.array(raw, copy=True)


def stack_optional(sample, out_key, refs, src_key, decoder):
    """Decode+stack `src_key` over `refs` into `sample[out_key]` if the
    current frame carries it. All refs must carry it; otherwise raise.
    """
    if refs[-1].get(src_key) is None:
        return
    arr = np.stack([decoder(f[src_key]) for f in refs], axis=0)
    if arr.shape[0] != len(refs):
        raise ValueError(f"{src_key}: got {arr.shape[0]} frames, expected {len(refs)}")
    sample[out_key] = arr


def materialize_sample_media(sample):
    """Decode RGB/depth from frame refs; absent keys are skipped silently.
    Which bytes arrive is gated upstream by ``select_files``."""
    image_refs = sample.pop("image_frame_refs", None)
    if image_refs is not None:
        stack_optional(sample, "image", image_refs, "image.jpg", decode_image_bytes)
        stack_optional(sample, "depth", image_refs, "depth.npy", decode_depth_bytes)
        stack_optional(sample, "chest_image", image_refs, "chest_image.jpg", decode_image_bytes)
        stack_optional(sample, "chest_depth", image_refs, "chest_depth.npy", decode_depth_bytes)

    future_refs = sample.pop("future_frame_refs", None)
    if future_refs is not None:
        stack_optional(sample, "future_frames", future_refs, "image.jpg", decode_image_bytes)
        stack_optional(sample, "chest_future_frames", future_refs, "chest_image.jpg", decode_image_bytes)

    return sample


def should_emit_window(frame, config):
    """DAgger Rule 1: False iff frame's high_quality=0 (missing key → 1)."""
    if not config.dagger_quality_filter:
        return True
    return int(frame["meta.json"].get("high_quality", 1)) == 1


def sliding_window_compose(src, config):
    """Sliding window over episode frames (streaming, not per-episode
    buffering).  Assumes shard order is contiguous within an episode;
    yields as soon as ``future_size`` future frames are buffered and
    clamps the tail at episode boundary.

    Failures inside ``build_sample_from_window`` are re-raised with the
    locator of ``buf[0]`` (the frame being windowed), and meta-access
    failures with the locator of the just-fetched ``sample``, so the
    error message points at the actual bad data rather than a buffered
    neighbour.
    """
    buf = collections.deque()
    past = collections.deque(maxlen=config.past_size)
    cur_ep = None

    def build_window():
        # ``buf[0]`` is the current frame the window is centered on,
        # captured before yielding so the closure stays cheap.
        current = buf[0]
        try:
            return build_sample_from_window(buf, past, config)
        except Exception as e:
            raise RuntimeError(
                f"data error on {build_sample_context(current)}"
            ) from e

    for sample in src:
        try:
            meta = sample["meta.json"]
            ep_key = (meta.get("dataset_name", ""), meta["episode_index"])
        except Exception as e:
            raise RuntimeError(
                f"data error on {build_sample_context(sample)}"
            ) from e

        if ep_key != cur_ep:
            # Episode boundary: flush with clamped actions.
            while buf:
                if should_emit_window(buf[0], config):
                    yield build_window()
                past.append(buf.popleft())
            past.clear()
            cur_ep = ep_key

        buf.append(sample)

        if len(buf) > config.future_size:
            if should_emit_window(buf[0], config):
                yield build_window()
            past.append(buf.popleft())

    while buf:
        if should_emit_window(buf[0], config):
            yield build_window()
        past.append(buf.popleft())


def no_split(src):
    """Identity splitter: yield all shards to every worker/node."""
    yield from src


def resolve_shuffle_initial(shuffle_buffer: int | None, shuffle_initial: int | None) -> int:
    """Cap WebDataset shuffle warmup so low keep_ratio does not stall startup."""
    if not shuffle_buffer or shuffle_buffer <= 0:
        return 0
    if shuffle_initial is None:
        return int(shuffle_buffer)
    return max(1, min(int(shuffle_initial), int(shuffle_buffer)))


def build_select_files(load_image: bool, load_depth: bool, load_chest: bool):
    """Allow-list predicate for ``wds.WebDataset(select_files=...)``.
    ``meta.json`` + ``lowdim.npy`` always pass; other VLA members gated
    by the three flags.  Uses webdataset's own ``base_plus_ext`` so the
    key/suffix split exactly matches how wds groups tar members.
    """
    allowed = {"meta.json", "lowdim.npy"}
    if load_image:
        allowed.add("image.jpg")
    if load_depth:
        allowed.add("depth.npy")
    if load_chest:
        allowed.add("chest_image.jpg")
    if load_depth and load_chest:
        allowed.add("chest_depth.npy")

    def predicate(fname: str) -> bool:
        _, suffix = base_plus_ext(fname)
        # base_plus_ext returns (None, None) for unsplittable names;
        # let wds handle those itself instead of dropping silently.
        return suffix is None or suffix in allowed
    return predicate


def build_wds_pipeline(shard_urls, config=None,
                       load_image=True, load_depth=False, load_chest=False,
                       preprocess_fn=None, shuffle_buffer=16384, shuffle_initial=None,
                       mode='train',
                       use_sliding_window=True,
                       include_post_stages=True,
                       keep_ratio: float = 1.0,
                       *, checker):
    """Build a WebDataset pipeline for a single dataset.

    Train: resampled infinite stream with shard-level shuffle.
    Val: finite single-pass, deterministic, no shuffle.

    ``include_post_stages=False`` stops after sliding-window compose so
    ``build_blended_dataset`` can share one shuffle + materialize stack
    across RandomMix.

    Args:
        shard_urls: tar path(s), braceexpand pattern, or list of globs.
        config: WindowConfig (defaults if None).
        load_image / load_depth / load_chest: tar-level modality gates.
            load_chest=True pulls chest_image; combined with load_depth
            also pulls chest_depth. Chest extrinsic/intrinsic slices
            are meta-driven (independent of load_chest).
        preprocess_fn: optional final map(sample) -> sample.
        shuffle_buffer: sample-level buffer (train only).
        shuffle_initial: number of kept samples to preload before yielding
            from shuffle. Keep this much smaller than shuffle_buffer when
            keep_ratio < 1.0 to avoid distributed startup stalls.
        mode: 'train' or 'val'.
        use_sliding_window: VLA=True, VLM=False. VLM path skips
            ``select_files`` since image_N.jpg has variable N.
        include_post_stages: if False, skip shuffle/materialize/preprocess.
        keep_ratio: Bernoulli pre-shuffle keep probability (train only).
            Lower = more shard diversity, higher IO.
            Ref: DreamZero shard_sampling_rate.
        checker: required ``DataChecker``. Every per-sample stage is
            wrapped via ``attach_sample_ctx`` so DataSkipError is logged
            and other failures carry sample locator info.
    """
    assert 0.0 < keep_ratio <= 1.0, f"keep_ratio must be in (0, 1], got {keep_ratio}"

    if config is None:
        config = WindowConfig()

    shard_urls, shard_patterns_metadata = expand_shard_patterns(shard_urls)
    assert shard_urls, f"No shards found: {shard_patterns_metadata}"

    is_train = (mode == 'train')
    # VLM uses image_N.jpg with variable N; skip tar-level filter.
    select_files = (
        build_select_files(load_image, load_depth, load_chest)
        if use_sliding_window else None
    )

    # resampled=True's ResampledShards already gives per-worker/node
    # independent shard sequences, so explicit splitters are redundant in
    # train mode. Ref: webdataset/shardlists.py::ResampledShards.__iter__
    stages = [
        wds.WebDataset(
            shard_urls,
            shardshuffle=False,
            nodesplitter=no_split if is_train else wds.split_by_node,
            workersplitter=no_split if is_train else wds.shardlists.split_by_worker,
            resampled=is_train,
            empty_check=False,
            select_files=select_files,
        ),
        # Eager meta/lowdim decode; media stays as bytes through shuffle.
        attach_sample_ctx(decode_sample_fields, checker=checker),
    ]

    if use_sliding_window:
        stages.append(lambda src: sliding_window_compose(src, config))

    # Drop pre-materialize so dropped windows skip JPEG decode.
    if is_train and keep_ratio < 1.0:
        keep_threshold = keep_ratio
        stages.append(wds.select(lambda _s: random.random() < keep_threshold))

    if include_post_stages:
        # Shuffle holds lightweight window descriptors (frame refs), not
        # decoded images.
        if is_train and shuffle_buffer and shuffle_buffer > 0:
            stages.append(wds.shuffle(
                shuffle_buffer, 
                initial=resolve_shuffle_initial(shuffle_buffer, shuffle_initial), 
            ))

        if use_sliding_window:
            stages.append(attach_sample_ctx(materialize_sample_media, checker=checker))
        else:
            stages.append(attach_sample_ctx(decode_media_fields, checker=checker))

        if preprocess_fn is not None:
            stages.append(attach_sample_ctx(preprocess_fn, checker=checker))

    return wds.DataPipeline(*stages)


def build_blended_dataset(datasets_config, config=None,
                          load_image=True, load_depth=False, load_chest=False,
                          preprocess_fn=None, shuffle_buffer=16384, shuffle_initial=None,
                          mode='train',
                          use_sliding_window=True,
                          keep_ratio: float = 1.0,
                          *, checker):
    """Build a blended dataset from multiple WebDataset sources.

    Train: per-subset pipelines mixed via RandomMix, then one shared
    shuffle + materialize + preprocess (memory independent of N subsets).
    Val: per-subset pipelines concatenated for single-pass eval.

    Args:
        datasets_config: list of {"shard_urls": ..., "weight": ...}.
        config: WindowConfig (defaults if None).
        load_image / load_depth / load_chest: see ``build_wds_pipeline``.
        preprocess_fn: optional final map(sample) -> sample.
        shuffle_buffer: sample-level buffer (train only).
        shuffle_initial: number of kept samples to preload before yielding
            from the train shuffle stage.
        mode: 'train' or 'val'.
        use_sliding_window: VLA=True, VLM=False.
        keep_ratio: forwarded per-subset; RandomMix weights are invariant
            because every subset is thinned by the same factor.
        checker: required ``DataChecker``; forwarded to per-subset
            pipelines and used to wrap post-mix stages so failures carry
            sample locator info.
    """
    if config is None:
        config = WindowConfig()

    is_train = (mode == 'train')

    subsets = []
    weights = []
    for c in datasets_config:
        urls = c["shard_urls"]
        # Train: subsets emit raw samples; shuffle/materialize/preprocess
        # run once after RandomMix.
        pipe = build_wds_pipeline(
            urls, config,
            load_image=load_image,
            load_depth=load_depth,
            load_chest=load_chest,
            preprocess_fn=preprocess_fn if not is_train else None,
            shuffle_buffer=shuffle_buffer,
            shuffle_initial=shuffle_initial,
            mode=mode,
            use_sliding_window=use_sliding_window,
            include_post_stages=not is_train,
            keep_ratio=keep_ratio,
            checker=checker,
        )
        subsets.append(pipe)
        weights.append(c.get("weight", 1.0))

    assert subsets, "No shards found across all datasets."

    if not is_train:
        def chain_pipelines():
            for pipe in subsets:
                yield from pipe
        return chain_pipelines()

    # RandomMix is an IterableDataset, not FluidInterface; wrap in
    # DataPipeline to append post-mix stages.
    mixed = subsets[0] if len(subsets) == 1 else wds.RandomMix(subsets, weights, longest=False)

    stages = [mixed]
    if shuffle_buffer and shuffle_buffer > 0:
        stages.append(wds.shuffle(
            shuffle_buffer,
            initial=resolve_shuffle_initial(shuffle_buffer, shuffle_initial),
        ))

    if use_sliding_window:
        stages.append(attach_sample_ctx(materialize_sample_media, checker=checker))
    else:
        stages.append(attach_sample_ctx(decode_media_fields, checker=checker))

    if preprocess_fn is not None:
        stages.append(attach_sample_ctx(preprocess_fn, checker=checker))

    return wds.DataPipeline(*stages)
