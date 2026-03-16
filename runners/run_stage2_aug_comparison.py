# runners/run_stage2_aug_comparison.py
# Run Stage-2 augmentation comparison for a given (model, dataset).
# Stage-2 LR is loaded from configs/protocol/stage2_selected_lrs.yaml.
# Stage-2 epochs are loaded from configs/protocol/stage2_epochs.yaml.

import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

STAGE2_BASE_CFG = ROOT / "configs" / "base" / "stage2.yaml"
DATASET_CFG_DIR = ROOT / "configs" / "datasets"
MODEL_CFG_DIR = ROOT / "configs" / "models"
AUG_CFG_DIR = ROOT / "configs" / "augs"
STAGE2_LR_CFG = ROOT / "configs" / "protocol" / "stage2_selected_lrs.yaml"
STAGE2_EPOCHS_CFG = ROOT / "configs" / "protocol" / "stage2_epochs.yaml"


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_stage2_lrs():
    return load_yaml(STAGE2_LR_CFG)


def load_stage2_epochs():
    return load_yaml(STAGE2_EPOCHS_CFG)


def discover_augs():
    augs = []
    for p in sorted(AUG_CFG_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        augs.append((p.stem, str(p)))
    return augs


def get_stage2_lr(model: str, dataset: str) -> float:
    lr_all = load_stage2_lrs()

    if model not in lr_all:
        raise KeyError(f"No Stage-2 LR block found for model: {model}")
    if dataset not in lr_all[model]:
        raise KeyError(f"No Stage-2 LR found for model={model}, dataset={dataset}")

    lr = lr_all[model][dataset]
    if lr is None:
        raise ValueError(f"Stage-2 LR is None for model={model}, dataset={dataset}")

    return float(lr)


def get_stage2_epochs(model: str, dataset: str) -> int:
    epochs_all = load_stage2_epochs()

    if model not in epochs_all:
        raise KeyError(f"No Stage-2 epoch block found for model: {model}")
    if dataset not in epochs_all[model]:
        raise KeyError(f"No Stage-2 epoch found for model={model}, dataset={dataset}")

    epochs = epochs_all[model][dataset]
    if epochs is None:
        raise ValueError(f"Stage-2 epochs is None for model={model}, dataset={dataset}")

    return int(epochs)


def build_cmd(dataset: str, model: str, aug_name: str, aug_yaml: str):
    ds_yaml = DATASET_CFG_DIR / f"{dataset}.yaml"
    model_yaml = MODEL_CFG_DIR / f"{model}.yaml"

    if not ds_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {ds_yaml}")
    if not model_yaml.is_file():
        raise FileNotFoundError(f"Model config not found: {model_yaml}")

    lr = get_stage2_lr(model, dataset)
    epochs = get_stage2_epochs(model, dataset)

    cmd = [
        "python",
        str(ROOT / "main.py"),
        str(STAGE2_BASE_CFG),
        str(ds_yaml),
        str(model_yaml),
        str(aug_yaml),
        "--override",
        "data.use_val_split=false",
        "early_stopping.enabled=false",
        "wandb.project=stage2-aug-comparison-v4",
        f"wandb.group={model}-{dataset}",
        f"train.lr={lr}",
        f"train.epochs={epochs}",
        f"train.exp_id=S2_{aug_name}",
    ]
    return cmd


def run_one(dataset: str, model: str, aug_name: str, aug_yaml: str, idx: int, total: int):
    cmd = build_cmd(dataset, model, aug_name, aug_yaml)

    print("=" * 80)
    print(f"[{idx}/{total}] [Stage-2] model={model} dataset={dataset} aug={aug_name}")
    print(f"Config: {aug_yaml}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 3:
        print("Usage: python runners/run_stage2_aug_comparison.py <dataset> <model> [aug_name]")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    model = sys.argv[2].lower()
    requested_aug = sys.argv[3].lower() if len(sys.argv) > 3 else None

    allowed_datasets = ["cifar10", "cifar100", "dermamnist", "pathmnist"]
    allowed_models = ["resnet18", "vit_b"]

    if dataset not in allowed_datasets:
        print(f"[ERROR] Unknown dataset: {dataset}")
        sys.exit(1)

    if model not in allowed_models:
        print(f"[ERROR] Unknown model: {model}")
        sys.exit(1)

    if not STAGE2_BASE_CFG.is_file():
        print(f"[ERROR] Missing Stage-2 base config: {STAGE2_BASE_CFG}")
        sys.exit(1)

    all_augs = discover_augs()
    if not all_augs:
        print(f"[ERROR] No augmentation yaml files found in {AUG_CFG_DIR}")
        sys.exit(1)

    all_aug_names = [name for name, _ in all_augs]
    print("\nAvailable augmentations:", ", ".join(all_aug_names))

    if requested_aug is not None:
        all_augs = [(name, y) for name, y in all_augs if name == requested_aug]
        if not all_augs:
            print(f"[ERROR] Aug '{requested_aug}' not found in configs/augs/")
            sys.exit(1)
        print(f"Will run ONLY augmentation: {requested_aug}")

    selected_lr = get_stage2_lr(model, dataset)
    selected_epochs = get_stage2_epochs(model, dataset)

    print(
        f"\nResolved Stage-2 contract: "
        f"model={model}, dataset={dataset}, lr={selected_lr}, epochs={selected_epochs}"
    )

    total = len(all_augs)
    print(f"Will run Stage-2 for model='{model}', dataset='{dataset}' with {total} augmentation(s).\n")

    for idx, (aug_name, aug_yaml) in enumerate(all_augs, start=1):
        run_one(dataset, model, aug_name, aug_yaml, idx, total)

    print("\nAll Stage-2 runs finished.\n")


if __name__ == "__main__":
    main()