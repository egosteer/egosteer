"""Release-spec fixtures: packed 74D Parquet, split metadata, four real videos.

Unlike the discarded implementation's tests, these fixtures use the on-disk
names/layout from the v1.11 PDF, and actual lossless HEVC gray12le depth.
"""

from copy import deepcopy
from itertools import islice
import json
import math
from pathlib import Path
import pickle
import shutil

import av
import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.dataset.lerobot_dataset import VLALeRobotStreamDataset, VLALowLevelLeRobotDataset
from src.dataset.lerobot_reader import LeRobotEpisodeReader
from src.dataset.lerobot_schema import unpack_motion
from src.dataset.lerobot_video import dequantize_depth_m
from src.dataset.normalizer_utils import get_normalizer
from src.dataset.vla_dataset import VLAWdsDataset
from src.dataset.wds_dataset import sliding_window_compose, materialize_sample_media


def shape_meta():
    return {"history_pad_mode": "repeat",
            "obs": {"rgb": {"horizon": 3, "stride": 2}, "depth": {"shape": [64, 64]},
                    "state": {"type": "fingertips", "horizon": 2, "stride": 1,
                              "wrist": {"shape": [18]}, "hand": {"shape": [30]}}},
            "action": {"horizon": 3, "stride": 1, "shape": [48], "pad_mode": "truncate"},
            "future_frame": {"horizon": 0, "stride": 2, "pad_mode": "repeat"}}


def motion(e, t):
    # Independently lay out the PDF's eight blocks. Distinct joints, sides,
    # translations and non-identity rotations expose slicing/reordering errors.
    result = list(np.arange(14, dtype=float) / 100) + [.2] * 6 + [.3] * 6
    tips = []
    for side in range(2):
        theta = .1 * (side + 1)
        rotation = np.array([[math.cos(theta), -math.sin(theta), 0],
                             [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        xyz = np.array([e * .1 + t * .002, side * .15, .5])
        result.extend(xyz)
        result.extend(rotation[:, :2].T.reshape(-1))
        for finger in range(5):
            tips.extend(xyz + rotation @ np.array([.01 * finger, .02, .03]))
    return np.array(result + tips, dtype=np.float32)


def write_video(path, frames, depth=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx265" if depth else "libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "gray12le" if depth else "yuv420p"
        stream.thread_count = 1
        stream.options = ({"x265-params": "lossless=1:log-level=error:pools=none:frame-threads=1:keyint=30:min-keyint=30:bframes=0"}
                          if depth else {"crf": "18", "g": "15", "bf": "0", "preset": "ultrafast"})
        for array in frames:
            if depth:
                # Write the code plane directly: conversion from gray16 would
                # rescale intensities instead of preserving 12-bit code values.
                frame = av.VideoFrame(64, 64, "gray12le")
                plane = frame.planes[0]
                pixels = np.frombuffer(plane, np.uint16).reshape(64, plane.line_size // 2)
                pixels[:] = 0
                pixels[:, :64] = array
            else:
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="session")
def release_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("lerobot_release")
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [74]},
        "action": {"dtype": "float32", "shape": [74]},
    }
    for cam in ("head", "chest"):
        features[f"observation.images.{cam}"] = {"dtype": "video", "shape": [64, 64, 3]}
        features[f"observation.images.{cam}_depth"] = {
            "dtype": "video", "shape": [64, 64, 1], "info": {
                "is_depth_map": True, "video.pix_fmt": "gray12le", "video.depth_min": .01,
                "video.depth_max": 10., "video.shift": 3.5, "video.use_log": True}}
    info = {"codebase_version": "v3.0", "fps": 30, "total_episodes": 4, "total_frames": 17,
            "splits": {"train": "0:3", "val": "3:4"}, "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (root / "meta/info.json").write_text(json.dumps(info))
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["Task A", "Task B"], name="task"))
    tasks.to_parquet(root / "meta/tasks.parquet")
    # Not a model-space normalizer; reader must not consume this.
    (root / "meta/stats.json").write_text(json.dumps({"observation.state": {"mean": [999.] * 74}}))
    rows, episodes, rgb = [], [], []
    for e, length in enumerate([7, 4, 1, 5]):
        start = len(rows)
        ep = {"episode_index": e, "length": length, "tasks": ["Task A" if e % 2 == 0 else "Task B"],
              "instructions": [f"Episode {e} instruction {i}" for i in range(12 if e == 0 else 2)],
              "split": "train" if e < 3 else "val", "dataset_from_index": start,
              "dataset_to_index": start + length, "data/chunk_index": 0, "data/file_index": 0,
              "stats/not_needed/mean": [9.] * 100}
        for cam in ("head", "chest"):
            ep[f"calibration.{cam}_intrinsics"] = [50., 0., 32., 0., 51., 32., 0., 0., 1.]
            ext = np.eye(4)
            if cam == "chest":
                ext[0, 3] = .12
            ep[f"calibration.{cam}_world2cam"] = ext.reshape(-1).tolist()
            for side in ("left", "right"):
                handeye = np.eye(4)
                handeye[1, 3] = 3.0 if side == "left" else -3.0
                ep[f"calibration.{cam}_cam_to_{side}_base"] = handeye.reshape(-1).tolist()
            prefix = f"videos/observation.images.{cam}"
            ep.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": 0,
                       f"{prefix}/from_timestamp": start / 30, f"{prefix}/to_timestamp": (start + length) / 30})
            # Depth has a different file AND nonzero offset from RGB, as in the
            # release where each feature rolls files independently.
            prefix = f"videos/observation.images.{cam}_depth"
            ep.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": e + 10,
                       f"{prefix}/from_timestamp": 2 / 30, f"{prefix}/to_timestamp": (2 + length) / 30})
            codes = [np.full((64, 64), 3000, np.uint16) for _ in range(2)]
            for k in range(length):
                code = np.full((64, 64), 1000 + e * 200 + k * 10 + (50 if cam == "chest" else 0), np.uint16)
                code[0, 0] = 0
                codes.append(code)
            write_video(root / f"videos/observation.images.{cam}_depth/chunk-000/file-{e+10:03d}.mp4", codes, depth=True)
        episodes.append(ep)
        for k in range(length):
            rows.append({"episode_index": e, "frame_index": k, "index": start + k,
                         "task_index": e % 2, "timestamp": np.float32(k / 30),
                         "observation.state": motion(e, k).tolist(),
                         # Fixture follows action[t] == state[t+1]; the final
                         # storage row holds the last state and is not an anchor.
                         "action": motion(e, min(k + 1, length - 1)).tolist()})
            frame = np.full((64, 64, 3), 40 + (start + k) * 7, np.uint8)
            frame[:, 32:] += 20
            rgb.append(frame)
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet", row_group_size=3)
    for part, group in enumerate((episodes[:2], episodes[2:])):
        pq.write_table(pa.Table.from_pylist(group), root / f"meta/episodes/chunk-000/file-{part:03d}.parquet")
    for cam in ("head", "chest"):
        write_video(root / f"videos/observation.images.{cam}/chunk-000/file-000.mp4", rgb)
    return root


def dataset(root, **kwargs):
    options = dict(root=str(root), shape_meta=shape_meta(), mode="val", split="train",
                   load_chest=True, load_depth=True, target_image_size=[64, 64],
                   drop_ratio=0, shuffle_buffer=4, shuffle_initial=2, return_dataset_info=True)
    options.update(kwargs)
    return VLALeRobotStreamDataset(**options)


def test_exact_74d_mapping_and_action_equals_next_state(release_root, monkeypatch):
    ds = dataset(release_root)
    read_lowdim = ds.reader.read_lowdim
    def read_state_only(e, indices, column):
        assert column == "observation.state", "training must derive targets from next state"
        return read_lowdim(e, indices, column)
    monkeypatch.setattr(ds.reader, "read_lowdim", read_state_only)
    raw = ds.read_window(0, 2, load_media=False)
    expected_state = motion(0, 2)
    expected_action = motion(0, 3)
    expected_wrist = np.r_[expected_state[26:29], expected_state[35:38], expected_state[29:35], expected_state[38:44]]
    np.testing.assert_array_equal(raw["wrist_state"][-1], expected_wrist)
    np.testing.assert_array_equal(raw["hand_action"][0], expected_action[44:74])
    np.testing.assert_array_equal(raw["wrist_action"][0, :3], motion(0, 3)[26:29])
    next_window = ds.read_window(0, 3, load_media=False)
    np.testing.assert_array_equal(raw["wrist_action"][0], next_window["wrist_state"][-1])
    np.testing.assert_array_equal(raw["hand_action"][0], next_window["hand_state"][-1])
    assert len(raw["instruction"]) == raw["instruction_num"] == 12
    assert raw["dataset_name"] == "Task A"
    # The 3m hand-eye translation must never be applied again to world-space FK.
    assert abs(raw["wrist_state"][0, 1]) < .001
    assert ds.reader.read_lowdim(0, [2], "observation.state").shape == (1, 74)


def test_split_and_metadata_projection(release_root):
    train, val = LeRobotEpisodeReader(release_root, "train"), LeRobotEpisodeReader(release_root, "val")
    assert [e["episode_index"] for e in train.episodes] == [0, 1, 2]
    assert [e["episode_index"] for e in val.episodes] == [3]
    assert "stats/not_needed/mean" not in train.episodes[0]
    assert "task_name" not in train.episodes[0] and "instruction_num" not in train.episodes[0]
    assert train.episodes[0]["calibration.head_intrinsics"].dtype == np.float64
    ds = dataset(release_root, mode="train", val_stride=2)
    validation = ds.get_validation_dataset()
    assert validation.root == ds.root and validation.split == "val"
    assert len(list(validation)) == 2  # k=0,2; final frame 4 has no next state


def test_real_hevc_depth_offsets_codes_and_units(release_root):
    reader = LeRobotEpisodeReader(release_root, "train", frame_cache_size=1)
    rgb = reader.read_media(1, "observation.images.head", [0, 2, 0])
    np.testing.assert_allclose(rgb.mean(axis=(1, 2, 3)), [99, 113, 99], atol=5)
    depth = reader.read_media(1, "observation.images.head_depth", [0, 2, 0])
    chest = reader.read_media(1, "observation.images.chest_depth", [0])
    def inverse(q):
        return math.exp(math.log(3.51) + q / 4095 * math.log(13.5 / 3.51)) - 3.5
    np.testing.assert_allclose(depth[:, 10, 10], [inverse(1200), inverse(1220), inverse(1200)], atol=1e-6)
    np.testing.assert_allclose(chest[0, 10, 10], inverse(1250), atol=1e-6)
    assert depth.dtype == np.float32 and np.all(depth[:, 0, 0] == 0)
    restored = pickle.loads(pickle.dumps(reader))
    assert not restored.video.frames and not restored.video.readers and not restored.row_groups
    np.testing.assert_array_equal(restored.read_media(1, "observation.images.head_depth", [0]), depth[:1])


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("action_pad", ["repeat", "truncate"])
@pytest.mark.parametrize("history_pad", ["repeat", "truncate"])
def test_model_sample_parity_with_wds(release_root, relative, action_pad, history_pad):
    meta = shape_meta()
    meta["history_pad_mode"] = history_pad
    meta["action"].update(stride=2, pad_mode=action_pad)
    meta["future_frame"]["horizon"] = 2
    ds = dataset(release_root, shape_meta=meta, use_relative_action=relative)
    ep, frames = ds.reader.episodes[0], []
    for k in range(6):
        lowdim = []
        # Independent reimplementation of the converter's 74D -> WDS 136D
        # packing; no call to the new unpack_motion when building the reference.
        for vector in (motion(0, k), motion(0, k + 1)):
            lowdim.extend(np.r_[vector[26:29], vector[35:38], vector[29:35], vector[38:44], vector[44:74]])
        for cam in ("head", "chest"):
            lowdim.extend(ep[f"calibration.{cam}_world2cam"].reshape(-1))
            lowdim.extend([50, 51, 32, 32])
        frame = {"meta.json": {"episode_index": 0, "cameras": ["head", "chest"],
                               "instruction": ep["instructions"], "instruction_num": 12},
                 "lowdim.npy": np.array(lowdim, np.float32)}
        for cam, prefix in (("head", ""), ("chest", "chest_")):
            frame[f"{prefix}image.jpg"] = ds.reader.read_media(0, f"observation.images.{cam}", [k])[0]
            frame[f"{prefix}depth.npy"] = ds.reader.read_media(0, f"observation.images.{cam}_depth", [k])[0]
        frames.append(frame)
    reference = VLAWdsDataset(wds_datasets=[], shape_meta=meta, mode="val", load_chest=True,
                              load_depth=True, target_image_size=[64, 64], use_relative_action=relative)
    for k, window in enumerate(sliding_window_compose(frames, ds.window_config)):
        expected = reference.sample_to_data(materialize_sample_media(window))
        actual = ds.sample_to_data(ds.read_window(0, k))
        # WDS reference omits the no-successor anchor. The new source can still
        # read the final observation for future RGB supervision, so compare BC
        # fields separately from future-frame boundary availability.
        for key in ("states", "actions", "actions_valid_mask", "n_states", "n_actions", "images",
                    "chest_images", "intrinsic"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, err_msg=f"k={k}, key={key}")
        future_ids = [min(k + 2, 6), min(k + 4, 6)]
        np.testing.assert_array_equal(actual["future_frames"],
                                      ds.reader.read_media(0, "observation.images.head", future_ids))
        np.testing.assert_allclose(actual["future_head_motion"], np.tile(np.eye(4).reshape(1, 16), (2, 1)), atol=1e-6)


def test_bc_requires_successor_and_no_rl_fields(release_root):
    samples = list(dataset(release_root))
    assert len(samples) == 9  # 6+3+0, no target fabricated for a final/singleton frame
    assert samples[-1]["n_actions"] == 1
    assert samples[-1]["actions_valid_mask"].sum() == 48
    for key in ("reward", "next_obs_sample", "bootstrap_mask", "bootstrap_discount", "is_intervention", "policy_version"):
        assert key not in samples[0]


def test_worker_partition_and_predecode_val_stride(release_root):
    ds = dataset(release_root, val_stride=3)
    calls = []
    def record(e, k):
        calls.append((e, k))
        return e, k
    ds.materialize = record
    assert list(ds.iter_samples(0, 2)) == [(0, 0), (0, 3)]
    assert list(ds.iter_samples(1, 2)) == [(1, 0)]
    assert len(calls) == 3  # discard before decoding, like current WDS val_stride
    assert list(ds.iter_samples(4, 5)) == []
    ds.mode = "train"
    with pytest.raises(ValueError, match="each training worker"):
        next(ds.iter_samples(4, 5))


def test_drop_and_shuffle_schedule_matches_rl(release_root):
    ds = dataset(release_root, mode="train", drop_ratio=.4, seed=3)
    calls = []
    def record(e, k):
        calls.append((e, k))
        return e, k
    ds.materialize = record
    outputs = list(islice(ds.iter_samples(0, 1), 12))
    candidates = []
    for round_idx in range(6):
        order = np.random.default_rng([3, round_idx, 0, 2]).permutation(2)
        rng = np.random.default_rng([3, round_idx, 0, 0])
        for e in order:
            for k in range([6, 3][e]):
                if rng.random() >= .4:
                    candidates.append((int(e), k))
    assert calls == candidates[:len(calls)]
    rng, source, buffer, expected = np.random.default_rng([3, 0, 1]), iter(candidates), [], []
    while len(expected) < 12:
        buffer.append(next(source))
        if len(buffer) < 4:
            buffer.append(next(source))
        pick = int(rng.integers(len(buffer)))
        expected.append(buffer[pick])
        buffer[pick] = buffer[-1]
        buffer.pop()
    assert outputs == expected


def test_normalizer_uses_train_real_rows_and_never_video(release_root, monkeypatch):
    ds = VLALowLevelLeRobotDataset(root=str(release_root), shape_meta=shape_meta(), use_relative_action=True,
                                  val_stride=999, drop_ratio=.99)
    def forbidden(*args, **kwargs):
        raise AssertionError("normalizer must not decode video")
    monkeypatch.setattr(LeRobotEpisodeReader, "read_media", forbidden)
    norm, stats = get_normalizer({"batch_size": 4, "num_workers": 0}, ds, return_metadata=True)
    assert stats["current_frames_scanned"] == 9
    assert stats["effective_rows"] == {"states": 18, "actions": 21}  # 15+6
    assert "actions" in norm.params_dict


def test_spawn_workers(release_root):
    loader = torch.utils.data.DataLoader(dataset(release_root), batch_size=None, num_workers=2,
                                         multiprocessing_context="spawn")
    samples = list(loader)
    assert len(samples) == 9
    assert sorted(int(s["episode_index"]) for s in samples) == [0]*6 + [1]*3


def test_hydra_integration_preserves_wds_config(release_root):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = Path(__file__).resolve().parents[1] / "src/config"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="experiment/egosteer_lerobot")
        original = hydra.compose(config_name="experiment/egosteer_qwen3_vl")
    cfg.lerobot_root = str(release_root)
    cfg.data.target_image_size = [64, 64]
    wrapped = hydra.utils.instantiate(cfg.dataset)
    assert wrapped.vla_dataset.val_stride == 1
    assert wrapped.vla_dataset.shuffle_initial == wrapped.vla_dataset.shuffle_buffer == 4096
    assert len(list(wrapped.get_validation_dataset())) == 4
    assert original.dataset.vla_dataset._target_.endswith("VLAWdsDataset")
    assert original.dataset.vla_dataset.val_stride == 1
    assert cfg.world_model.enabled == original.world_model.enabled


@pytest.mark.parametrize("problem", ["wrong_vector", "wrong_split", "missing_depth_params"])
def test_reject_draft_schema_and_inconsistent_release_metadata(release_root, tmp_path, problem):
    copy = tmp_path / "bad_release"
    shutil.copytree(release_root, copy)
    path = copy / "meta/info.json"
    info = json.loads(path.read_text())
    if problem == "wrong_vector":
        info["features"]["observation.state"]["shape"] = [48]
    elif problem == "wrong_split":
        info["splits"]["train"] = "0:4"  # last episode is explicitly val
    else:
        del info["features"]["observation.images.head_depth"]["info"]["video.shift"]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        LeRobotEpisodeReader(copy, "train")


def test_depth_linear_nondefault_metadata():
    # Guards against hardcoding the PDF's log defaults instead of reading info.
    codes = np.array([[0, 4095]], dtype=np.uint16)
    result = dequantize_depth_m(codes, depth_min=.2, depth_max=4., shift=1., use_log=False)
    np.testing.assert_allclose(result, [[0., 4.]])
