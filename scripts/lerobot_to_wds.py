#!/usr/bin/env python3
"""Convert an EgoSteer LeRobot v3 dataset into WebDataset shards for training.

Output layout (contract in data/data.md):

    <out>/train/shard-000000.tar ...      <out>/val/shard-000000.tar ...

Every frame is one sample with four members:

    episode_000123_frame_000045.image.jpg        head RGB, JPEG
    episode_000123_frame_000045.chest_image.jpg  chest RGB, JPEG
    episode_000123_frame_000045.lowdim.npy       float32[136], see to_lowdim()
    episode_000123_frame_000045.meta.json        instruction / instruction_num / episode_index / dataset_name / cameras

Episodes are shuffled with a fixed seed and packed *whole* into shards, so each
shard mixes tasks and no episode is ever split across two shards. The loader
relies on that: it cuts temporal windows at episode boundaries inside a shard.

Usage:
    python scripts/lerobot_to_wds.py --root <lerobot dataset> --out <wds dir> \
        [--frames-per-shard 1000] [--seed 0] [--workers 32] [--jpeg-quality 95] [--part 0/1]

Re-running skips shards that already exist, so an interrupted run can resume.
To spread the work over several machines that share the output directory, give
each one a different --part k/N: machine k writes the shards whose index % N == k.
Requires numpy, pyarrow, av (PyAV) and Pillow.
"""
import argparse
import io
import json
import os
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HEAD, CHEST = "observation.images.head", "observation.images.chest"
WORLD2CAM = {"head": "observation.camera.head_world2cam", "chest": "observation.camera.chest_world2cam"}


# ----------------------------------------------------------------------------- reading LeRobot

def load_episodes(root):
    """One dict per episode from meta/episodes/*.parquet, sorted by episode_index."""
    wanted = ["episode_index", "tasks", "length", "split", "instructions",
              "data/chunk_index", "data/file_index",
              "calibration/head_intrinsics", "calibration/chest_intrinsics"]
    for cam in (HEAD, CHEST):
        wanted += [f"videos/{cam}/chunk_index", f"videos/{cam}/file_index",
                   f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"]
    rows = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        rows += pq.read_table(path, columns=wanted).to_pylist()
    return sorted(rows, key=lambda r: r["episode_index"])


def read_lowdim_rows(root, info, ep):
    """(state, action, head_world2cam, chest_world2cam) float32 arrays for one episode.

    The dataset writes one parquet row group per episode, so only that row
    group is read instead of the whole file.
    """
    path = root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        first = pf.read_row_group(rg, columns=["episode_index"]).column(0)[0].as_py()
        if first == ep["episode_index"]:
            table = pf.read_row_group(rg, columns=["episode_index", "observation.state", "action", *WORLD2CAM.values()])
            break
    else:
        raise KeyError(f"episode {ep['episode_index']} not found in {path}")
    assert set(table.column("episode_index").to_pylist()) == {ep["episode_index"]}, "row group holds more than one episode"
    columns = [np.asarray(table.column(c).to_pylist(), dtype=np.float32) for c in ("observation.state", "action", *WORLD2CAM.values())]
    assert len(columns[0]) == ep["length"], f"episode {ep['episode_index']}: {len(columns[0])} rows, expected {ep['length']}"
    return columns


def decode_video(root, info, ep, cam):
    """Yield the RGB frames of one episode from the video file that stores it.

    Every episode starts on a keyframe, so seeking to from_timestamp and
    decoding until to_timestamp returns exactly the episode's frames.
    """
    path = root / info["video_path"].format(video_key=cam, chunk_index=ep[f"videos/{cam}/chunk_index"],
                                           file_index=ep[f"videos/{cam}/file_index"])
    t0, t1 = ep[f"videos/{cam}/from_timestamp"], ep[f"videos/{cam}/to_timestamp"]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type, stream.thread_count = "SLICE", 1
        container.seek(int(t0 / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < t0 - 1e-6:
                continue
            if t >= t1 - 1e-6:
                break
            yield frame.to_ndarray(format="rgb24")


# ----------------------------------------------------------------------------- sample encoding

def to_lowdim(state, action, head_world2cam, chest_world2cam, head_K, chest_K):
    """Map one frame of the 74-dim LeRobot state/action to the 136-dim lowdim vector.

    LeRobot layout: [arm_L(7) arm_R(7) hand_L(6) hand_R(6) wrist_L(9) wrist_R(9) tips_L(15) tips_R(15)]
    where wrist = [xyz(3) rot6d(6)] in the head-camera (world) frame.
    lowdim layout: wrist_state(18) hand_state(30) wrist_action(18) hand_action(30)
                   head_extrinsic(16) head_intrinsic(4) chest_extrinsic(16) chest_intrinsic(4)
    The extrinsics are the frame's world2cam matrices (4x4, row-major) as stored in the dataset.
    """
    def wrist(v):
        return np.concatenate([v[26:29], v[35:38], v[29:35], v[38:44]])   # L xyz, R xyz, L rot6d, R rot6d

    def intrinsic(K):   # K is a row-major 3x3 -> [fx, fy, cx, cy]
        return [K[0], K[4], K[2], K[5]]

    return np.concatenate([wrist(state), state[44:74], wrist(action), action[44:74],
                           head_world2cam, intrinsic(head_K), chest_world2cam, intrinsic(chest_K)]).astype(np.float32)


def jpeg_bytes(rgb, quality):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def npy_bytes(array):
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


# ----------------------------------------------------------------------------- shards

def plan_shards(episodes, frames_per_shard, seed):
    """{split: [[episode, ...], ...]}: shuffle episodes per split, then pack them whole."""
    rng = np.random.default_rng(seed)
    plan = {}
    for split in sorted({e["split"] for e in episodes}):
        pool = [e for e in episodes if e["split"] == split]
        pool = [pool[i] for i in rng.permutation(len(pool))]
        shards, current, frames = [], [], 0
        for ep in pool:
            if current and frames + ep["length"] > frames_per_shard:
                shards.append(current)
                current, frames = [], 0
            current.append(ep)
            frames += ep["length"]
        if current:
            shards.append(current)
        plan[split] = shards
    return plan


def write_shard(job):
    root, info, out_path, episodes, quality = job
    root, out_path = Path(root), Path(out_path)
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")   # unique per process, renamed when complete
    n_frames = 0
    with tarfile.open(tmp_path, "w") as tar:
        def add(name, data):
            member = tarfile.TarInfo(name)
            member.size, member.mtime = len(data), 0
            tar.addfile(member, io.BytesIO(data))

        for ep in episodes:
            state, action, head_world2cam, chest_world2cam = read_lowdim_rows(root, info, ep)
            meta = {"instruction": list(ep["instructions"]), "instruction_num": len(ep["instructions"]),
                    "episode_index": ep["episode_index"], "dataset_name": ep["tasks"][0], "cameras": ["head", "chest"]}
            meta_json = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            frames = zip(decode_video(root, info, ep, HEAD), decode_video(root, info, ep, CHEST))
            decoded = 0
            for t, (head, chest) in enumerate(frames):
                key = f"episode_{ep['episode_index']:06d}_frame_{t:06d}"
                add(key + ".image.jpg", jpeg_bytes(head, quality))
                add(key + ".chest_image.jpg", jpeg_bytes(chest, quality))
                add(key + ".lowdim.npy", npy_bytes(to_lowdim(state[t], action[t], head_world2cam[t], chest_world2cam[t],
                                                              ep["calibration/head_intrinsics"], ep["calibration/chest_intrinsics"])))
                add(key + ".meta.json", meta_json)
                decoded += 1
            assert decoded == ep["length"], f"episode {ep['episode_index']}: decoded {decoded} frames, expected {ep['length']}"
            n_frames += decoded
    tmp_path.rename(out_path)
    return out_path.name, len(episodes), n_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="LeRobot v3 dataset root")
    ap.add_argument("--out", required=True, type=Path, help="output directory; gets train/ and val/ subdirectories")
    ap.add_argument("--frames-per-shard", type=int, default=1000, help="target frames per shard (episodes are never split)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the episode shuffle")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--part", default="0/1", help="k/N: only write shards with index %% N == k (for multi-machine runs)")
    args = ap.parse_args()
    part, n_parts = map(int, args.part.split("/"))
    assert 0 <= part < n_parts, f"--part {args.part}: expected k/N with 0 <= k < N"

    info = json.loads((args.root / "meta" / "info.json").read_text())
    episodes = load_episodes(args.root)
    plan = plan_shards(episodes, args.frames_per_shard, args.seed)

    jobs = []
    for split, shards in plan.items():
        (args.out / split).mkdir(parents=True, exist_ok=True)
        index = {"seed": args.seed, "frames_per_shard": args.frames_per_shard,
                 "shards": {f"shard-{i:06d}.tar": [ep["episode_index"] for ep in eps] for i, eps in enumerate(shards)}}
        index_path = args.out / split / "index.json"          # shard -> episodes, for reproducibility
        if index_path.exists():                               # resuming: the existing shards must come from the same plan
            assert json.loads(index_path.read_text()) == index, f"{index_path} was written with different arguments; use a new --out"
        else:
            index_path.write_text(json.dumps(index, indent=1))
        for i, (name, eps) in enumerate(zip(index["shards"], shards)):
            if i % n_parts == part and not (args.out / split / name).exists():
                jobs.append((str(args.root), info, str(args.out / split / name), eps, args.jpeg_quality))
    total = sum(len(s) for s in plan.values())
    print(f"{len(episodes)} episodes -> {total} shards; this run (part {args.part}): {len(jobs)} to write", flush=True)

    start = time.time()
    with Pool(args.workers) as pool:
        for i, (name, n_eps, n_frames) in enumerate(pool.imap_unordered(write_shard, jobs), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {(time.time() - start) / 60:.1f} min  last: {name} {n_eps} episodes {n_frames} frames", flush=True)
    print("done")


if __name__ == "__main__":
    main()
