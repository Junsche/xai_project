# runners/run_stage1_lr_grid.py
# Run Stage-1 baseline learning-rate grid search for a given (model, dataset).
# LR grid is loaded from configs/protocol/stage1_lr_grid.yaml.
# Stage-1 epochs are loaded from configs/protocol/stage1_epochs.yaml.

import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

STAGE1_BASE_CFG = ROOT / "configs" / "base" / "stage1.yaml"
DATASET_CFG_DIR = ROOT / "configs" / "datasets"
MODEL_CFG_DIR = ROOT / "configs" / "models"
AUG_CFG = ROOT / "configs" / "augs" / "baseline.yaml"
STAGE1_LR_GRID_CFG = ROOT / "configs" / "protocol" / "stage1_lr_grid.yaml"
STAGE1_EPOCHS_CFG = ROOT / "configs" / "protocol" / "stage1_epochs.yaml"


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lr_grid():
    return load_yaml(STAGE1_LR_GRID_CFG)


def load_stage1_epochs():
    return load_yaml(STAGE1_EPOCHS_CFG)


def get_lr_grid(model: str, dataset: str):
    lr_grid_all = load_lr_grid()

    if model not in lr_grid_all:
        raise KeyError(f"No Stage-1 LR grid block found for model: {model}")
    if dataset not in lr_grid_all[model]:
        raise KeyError(f"No Stage-1 LR grid found for model={model}, dataset={dataset}")

    lr_values = lr_grid_all[model][dataset]
    if not isinstance(lr_values, list) or len(lr_values) == 0:
        raise ValueError(f"Invalid LR grid for model={model}, dataset={dataset}: {lr_values}")

    return lr_values


def get_stage1_epochs(model: str, dataset: str) -> int:
    epochs_all = load_stage1_epochs()

    if model not in epochs_all:
        raise KeyError(f"No Stage-1 epoch block found for model: {model}")
    if dataset not in epochs_all[model]:
        raise KeyError(f"No Stage-1 epoch found for model={model}, dataset={dataset}")

    epochs = epochs_all[model][dataset]
    if epochs is None:
        raise ValueError(f"Stage-1 epochs is None for model={model}, dataset={dataset}")

    return int(epochs)


def run(dataset: str, model: str) -> None:
    dataset = dataset.lower()
    model = model.lower()

    allowed_datasets = ["cifar10", "cifar100", "dermamnist", "pathmnist"]
    allowed_models = ["resnet18", "vit_b"]

    if dataset not in allowed_datasets:
        raise ValueError(f"Unknown dataset: {dataset}")
    if model not in allowed_models:
        raise ValueError(f"Unknown model: {model}")

    ds_yaml = DATASET_CFG_DIR / f"{dataset}.yaml"
    model_yaml = MODEL_CFG_DIR / f"{model}.yaml"

    if not ds_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {ds_yaml}")
    if not model_yaml.is_file():
        raise FileNotFoundError(f"Model config not found: {model_yaml}")
    if not AUG_CFG.is_file():
        raise FileNotFoundError(f"Augmentation config not found: {AUG_CFG}")
    if not STAGE1_BASE_CFG.is_file():
        raise FileNotFoundError(f"Stage-1 base config not found: {STAGE1_BASE_CFG}")

    lr_values = get_lr_grid(model, dataset)
    stage1_epochs = get_stage1_epochs(model, dataset)

    exp_ids = [f"C{i+1}" for i in range(len(lr_values))]
    grid = list(zip(lr_values, exp_ids))

    wandb_group = f"{model}-{dataset}-stage1"
    wandb_project = "stage1-lr-grid-v4"

    for lr, exp in grid:
        cmd = [
            "python",
            str(ROOT / "main.py"),
            str(STAGE1_BASE_CFG),
            str(ds_yaml),
            str(model_yaml),
            str(AUG_CFG),
            "--override",
            f"train.lr={lr}",
            f"train.epochs={stage1_epochs}",
            f"train.exp_id={exp}",
            f"wandb.project={wandb_project}",
            f"wandb.group={wandb_group}",
        ]

        print("=" * 80)
        print(
            f"[Stage-1] model={model}, dataset={dataset}, "
            f"lr={lr}, epochs={stage1_epochs}, exp_id={exp}"
        )
        print("Command:", " ".join(cmd))
        print("=" * 80)

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    model = sys.argv[2] if len(sys.argv) > 2 else "resnet18"
    run(dataset, model)