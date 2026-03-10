# runners/run_stage1_lr_grid.py
# Run Stage-1 baseline learning-rate grid search for a given dataset.

import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

STAGE1_BASE_CFG = ROOT / "configs" / "base" / "stage1.yaml"
DATASET_CFG_DIR = ROOT / "configs" / "datasets"
MODEL_CFG = ROOT / "configs" / "models" / "resnet18.yaml"
AUG_CFG = ROOT / "configs" / "augs" / "baseline.yaml"
LR_GRID_CFG = ROOT / "configs" / "protocol" / "stage1_lr_grid.yaml"


def load_lr_grid():
    with open(LR_GRID_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(dataset: str) -> None:
    dataset = dataset.lower()
    allowed = ["cifar10", "cifar100", "dermamnist", "pathmnist"]
    assert dataset in allowed, f"Unknown dataset: {dataset}"

    ds_yaml = DATASET_CFG_DIR / f"{dataset}.yaml"
    if not ds_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {ds_yaml}")

    if not MODEL_CFG.is_file():
        raise FileNotFoundError(f"Model config not found: {MODEL_CFG}")

    lr_grid_all = load_lr_grid()
    if dataset not in lr_grid_all:
        raise KeyError(f"No LR grid found for dataset: {dataset}")

    lr_values = lr_grid_all[dataset]
    exp_ids = [f"C{i+1}" for i in range(len(lr_values))]
    grid = list(zip(lr_values, exp_ids))

    for lr, exp in grid:
        cmd = [
            "python",
            str(ROOT / "main.py"),
            str(STAGE1_BASE_CFG),
            str(ds_yaml),
            str(MODEL_CFG),
            str(AUG_CFG),
            "--override",
            f"train.lr={lr}",
            f"train.exp_id={exp}",
            "wandb.project=stage1-lr-grid-v4",
            "wandb.group=stage1",
        ]

        print("=" * 80)
        print(f"[Stage-1] dataset={dataset}, lr={lr}, exp_id={exp}")
        print("Command:", " ".join(cmd))
        print("=" * 80)

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    run(ds)