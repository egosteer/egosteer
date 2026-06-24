#!/usr/bin/env python3
"""Convert an FSDP2 DCP checkpoint into a single .pt file (model weights only).

Supports two checkpoint layouts:
  1. Native DCP dir (contains .metadata + __*_*.distcp files)
  2. Accelerate FSDP dir (contains pytorch_model_fsdp_0/ subdir)

Typical usage:

    python src/utils/convert_fsdp_checkpoint.py \
        --checkpoint outputs/<your-run>/step_checkpoints/update_step_65000

    python src/utils/convert_fsdp_checkpoint.py \
        --checkpoint outputs/<your-run>/update_step_130000 \
        --output /tmp/model.pt
"""
import argparse
import pathlib

import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    parser = argparse.ArgumentParser(
        description="Convert an FSDP2 DCP checkpoint into a single .pt file (model weights only)."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint root directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt path. Defaults to <checkpoint>/model.pt.",
    )
    parser.add_argument(
        "--keep-frozen-teacher",
        action="store_true",
        help="Keep frozen DINOv3 teacher weights. By default they are dropped: "
             "they are third-party frozen weights, unused at inference, and "
             "rebuilt via from_pretrained when the model is constructed.",
    )
    args = parser.parse_args()

    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")

    output = (
        pathlib.Path(args.output).expanduser().resolve()
        if args.output
        else ckpt / "model.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # Detect layout
    fsdp_subdir = ckpt / "pytorch_model_fsdp_0"
    if fsdp_subdir.exists():
        dcp_dir = fsdp_subdir
    elif (ckpt / ".metadata").exists():
        dcp_dir = ckpt
    else:
        raise FileNotFoundError(
            f"Neither .metadata nor pytorch_model_fsdp_0/ found in {ckpt}. "
            "Is this a valid FSDP checkpoint?"
        )

    print(f"Loading DCP checkpoint from {dcp_dir} ...")

    # dcp.load requires a pre-populated state_dict with the right structure,
    # which we don't have without instantiating the model. Use dcp_to_torch_save
    # to convert the full DCP dir into a single .pt, then extract model weights.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    dcp_to_torch_save(str(dcp_dir), tmp_path)
    full_state = torch.load(tmp_path, map_location="cpu", weights_only=False)
    pathlib.Path(tmp_path).unlink()

    # Extract model weights only (skip optimizer, scheduler, etc.)
    # Native DCP layout: {"app": {"model": ..., "optimizer": ..., ...}}
    if "app" in full_state and "model" in full_state["app"]:
        model_state = full_state["app"]["model"]
    elif "model" in full_state:
        model_state = full_state["model"]
    else:
        print("[WARN] Could not find 'model' key; saving full state_dict.")
        model_state = full_state

    if not args.keep_frozen_teacher:
        before = len(model_state)
        # startswith is safe: torch.compile only wraps inner ModuleLists, so the
        # "_orig_mod." marker (if any) appears mid-key, never at the front.
        model_state = {
            k: v for k, v in model_state.items()
            if not k.startswith("frozen_teacher.")
        }
        dropped = before - len(model_state)
        print(f"Dropped {dropped} frozen_teacher.* tensors ({before} -> {len(model_state)} params).")

    print(f"Saving {len(model_state)} parameters to {output} ...")
    torch.save(model_state, str(output))
    print(f"Done. File size: {output.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
