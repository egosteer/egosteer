"""Fit the model-space normalizer from a lowdim-only LeRobot train split."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle

import hydra
from omegaconf import OmegaConf

from src.dataset.normalizer_utils import get_normalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/experiment/egosteer_lerobot.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. lerobot_root=/path/to/lerobot_dataset")
    args = parser.parse_args()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)
    config = Path(args.config).resolve()
    with hydra.initialize_config_dir(config_dir=str(config.parent.parent), version_base=None):
        cfg = hydra.compose(config_name=f"{config.parent.name}/{config.stem}", overrides=args.overrides)
    data_cfg = OmegaConf.to_container(cfg.dataset.vla_dataset, resolve=True)
    data_cfg["_target_"] = "src.dataset.lerobot.vla_dataset.VLALowLevelLeRobotDataset"
    dataset = hydra.utils.instantiate(data_cfg)
    normalizer, summary = get_normalizer(
        {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": False},
        dataset, return_metadata=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "normalizer.pkl").open("wb") as file:
        pickle.dump(normalizer, file)
    metadata = {"format": "EgoSteer-RealWorld v1.11 / LeRobot v3.0", "root": dataset.root,
                "split": dataset.split, "shape_meta": OmegaConf.to_container(cfg.data.shape_meta, resolve=True),
                "use_relative_action": dataset.use_relative_action, "scan_summary": summary}
    (output / "normalizer.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output / 'normalizer.pkl'} ({summary['current_frames_scanned']} train anchors)")


if __name__ == "__main__":
    main()
