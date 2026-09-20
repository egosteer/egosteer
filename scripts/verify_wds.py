#!/usr/bin/env python3
"""Check that a WebDataset directory satisfies the loader contract in data/data.md.

For every shard: each frame has exactly image.jpg, chest_image.jpg, lowdim.npy,
meta.json in that order; frames of an episode are contiguous and numbered
0..N-1; meta fields are valid; lowdim is float32[136]. Across shards: no
episode appears in more than one shard, and each shard holds exactly the
episodes recorded for it in <split>/index.json (written by lerobot_to_wds.py).
With --root, every episode of the LeRobot dataset must be present with the
same number of frames.

Usage: python scripts/verify_wds.py --wds <wds dir> [--root <lerobot dataset>] [--workers 32]
"""
import argparse
import io
import json
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MEMBERS = ["image.jpg", "chest_image.jpg", "lowdim.npy", "meta.json"]


def check_shard(path):
    """Return (split, name, {episode_index: n_frames}, error) for one shard; error is None if it passes."""
    try:
        return path.parent.name, path.name, _check_shard(path), None
    except Exception as e:                       # AssertionError, tarfile.ReadError, ...
        return path.parent.name, path.name, {}, f"{type(e).__name__}: {e}"


def _check_shard(path):
    episodes, current, expected_frame, suffixes = {}, None, 0, []
    with tarfile.open(path) as tar:
        for member in tar:
            key, suffix = member.name.split(".", 1)
            suffixes.append(suffix)
            if suffix != "meta.json":
                if suffix == "lowdim.npy":
                    vec = np.load(io.BytesIO(tar.extractfile(member).read()))
                    assert vec.dtype == np.float32 and vec.shape == (136,), f"{path}:{member.name} lowdim {vec.dtype}{vec.shape}"
                continue
            assert suffixes == MEMBERS, f"{path}:{key} members {suffixes}"
            suffixes = []
            meta = json.load(tar.extractfile(member))
            episode, frame = int(key.split("_")[1]), int(key.split("_")[3])
            assert meta["episode_index"] == episode, f"{path}:{key} meta episode_index {meta['episode_index']}"
            assert meta["instruction_num"] == len(meta["instruction"]) > 0, f"{path}:{key} instruction_num mismatch"
            assert all(isinstance(s, str) and s.strip() for s in meta["instruction"]), f"{path}:{key} empty instruction"
            assert meta["cameras"] == ["head", "chest"], f"{path}:{key} cameras {meta['cameras']}"
            if episode != current:
                assert episode not in episodes, f"{path}: episode {episode} is not contiguous"
                current, expected_frame = episode, 0
            assert frame == expected_frame, f"{path}:{key} expected frame {expected_frame}"
            expected_frame += 1
            episodes[episode] = expected_frame
    assert not suffixes, f"{path}: trailing members {suffixes}"
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds", required=True, type=Path)
    ap.add_argument("--root", type=Path, help="LeRobot dataset to compare against")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    shards = sorted(args.wds.glob("*/shard-*.tar"))
    index = {}                                  # (split, shard) -> episodes planned for it
    for p in args.wds.glob("*/index.json"):
        index.update({(p.parent.name, name): set(eps) for name, eps in json.load(open(p))["shards"].items()})
    seen = {}                                   # episode -> (split, shard)
    per_split = {}
    bad = []
    with Pool(args.workers) as pool:
        for split, name, episodes, error in pool.imap_unordered(check_shard, shards):
            if error:
                bad.append(f"{split}/{name}")
                print(f"BAD {error}", flush=True)
                continue
            if (split, name) in index and set(episodes) != index[(split, name)]:
                bad.append(f"{split}/{name}")
                print(f"BAD {split}/{name}: holds episodes {sorted(episodes)}, index.json says {sorted(index[(split, name)])}", flush=True)
                continue
            for ep, n in episodes.items():
                if ep in seen:
                    bad.append(f"{split}/{name}")
                    print(f"BAD episode {ep} in both {seen[ep]} and {split}/{name}", flush=True)
                seen[ep] = (split, name)
                per_split.setdefault(split, {})[ep] = n
    if bad:
        raise SystemExit(f"{len(bad)} bad shard(s): " + " ".join(sorted(bad)))
    for split, eps in sorted(per_split.items()):
        print(f"{split}: {len([s for s in shards if s.parent.name == split])} shards, {len(eps)} episodes, {sum(eps.values())} frames")

    if args.root:
        rows = []
        for p in sorted((args.root / "meta" / "episodes").rglob("*.parquet")):
            rows += pq.read_table(p, columns=["episode_index", "length", "split"]).to_pylist()
        mismatch = [r for r in rows if per_split.get(r["split"], {}).get(r["episode_index"]) != r["length"]]
        for r in mismatch:
            print(f"BAD episode {r['episode_index']} ({r['split']}): wds {per_split.get(r['split'], {}).get(r['episode_index'])} frames, lerobot {r['length']}")
        if mismatch or len(rows) != len(seen):
            raise SystemExit(f"{len(mismatch)} episode(s) missing or wrong length; {len(rows)} episodes in lerobot, {len(seen)} in wds")
        print(f"matches {args.root}: {len(rows)} episodes, {sum(r['length'] for r in rows)} frames")
    print("ok")


if __name__ == "__main__":
    main()
