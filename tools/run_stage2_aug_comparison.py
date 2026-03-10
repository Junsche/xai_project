"""
Stage-2 augmentation comparison runner.

This script runs Stage-2 experiments for one dataset.
It loops over all augmentation configs in `configs/augs/*.yaml` and calls `main.py` for each one,
using Stage-1 selected hyperparameters (lr, epochs).
"""

import subprocess
import sys
from pathlib import Path

BASE_CFG = "configs/_base-comparison.yaml"

# Stage-1 selected hyperparameters per dataset
DATASETS = {
    "cifar10": {
        "yaml": "configs/datasets/cifar10.yaml",
        "lr": 0.01,
        "epochs": 100,
        "group": "cifar10",
    },
    "cifar100": {
        "yaml": "configs/datasets/cifar100.yaml",
        "lr": 0.001,
        "epochs": 100,
        "group": "cifar100",
    },
    "dermamnist": {
        "yaml": "configs/datasets/dermamnist.yaml",
        "lr": 0.005,
        "epochs": 50,
        "group": "dermamnist",
    },
    "pathmnist": {
        "yaml": "configs/datasets/pathmnist.yaml",
        "lr": 0.005,
        "epochs": 50,
        "group": "pathmnist",
    },
}


def discover_augs():
    """
    Discover all augmentation configs automatically by scanning `configs/augs/*.yaml`.
    Returns a list of (aug_name, aug_yaml_path).
    """
    aug_dir = Path("configs/augs")
    augs = []
    for p in sorted(aug_dir.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        augs.append((p.stem, str(p)))
    return augs


def build_cmd(ds_cfg: dict, aug_name: str, aug_yaml: str):
    """
    Build the command line for calling main.py for one (dataset, augmentation).
    """
    return [
        "python",
        "main.py",
        BASE_CFG,
        ds_cfg["yaml"],
        aug_yaml,
        "--override",
        # Stage-2 protocol constraints (fair comparison)
        "data.use_val_split=false",
        "early_stopping.enabled=false",
        # W&B metadata
        "wandb.project=stage2-aug-comparison-v3.1",
        f"wandb.group={ds_cfg['group']}",
        # Stage-1 selected hyperparameters
        f"train.lr={ds_cfg['lr']}",
        f"train.epochs={ds_cfg['epochs']}",
        # Stage-2 exp_id naming
        f"train.exp_id=S2_{aug_name}",
    ]


def run_one(ds_name: str, ds_cfg: dict, aug_name: str, aug_yaml: str, idx: int, total: int):
    """
    Run one experiment and print progress and the exact command.
    """
    print("=" * 80)
    print(f"[{idx}/{total}] [Stage-2] dataset={ds_name} aug={aug_name}")
    print(f"Config: {aug_yaml}")
    cmd = build_cmd(ds_cfg, aug_name, aug_yaml)
    print("Command:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/run_comparison.py <dataset> [aug_name]")
        sys.exit(1)

    ds_name = sys.argv[1].lower()
    if ds_name not in DATASETS:
        print(f"[ERROR] Unknown dataset: {ds_name}")
        sys.exit(1)

    requested_aug = sys.argv[2].lower() if len(sys.argv) > 2 else None
    ds_cfg = DATASETS[ds_name]

    all_augs = discover_augs()
    if not all_augs:
        print("[ERROR] No augmentation yaml files found in configs/augs")
        sys.exit(1)

    all_aug_names = [name for name, _ in all_augs]
    print("\nAvailable augmentations:", ", ".join(all_aug_names))

    if requested_aug is not None:
        all_augs = [(name, y) for name, y in all_augs if name == requested_aug]
        if not all_augs:
            print(f"[ERROR] Aug '{requested_aug}' not found in configs/augs/")
            sys.exit(1)
        print(f"Will run ONLY augmentation: {requested_aug}")

    total = len(all_augs)
    print(f"\nWill run Stage-2 for dataset '{ds_name}' with {total} augmentation(s).\n")

    for idx, (aug_name, aug_yaml) in enumerate(all_augs, start=1):
        run_one(ds_name, ds_cfg, aug_name, aug_yaml, idx, total)

    print("\nAll Stage-2 runs finished.\n")


if __name__ == "__main__":
    main()
