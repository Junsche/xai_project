# runners/run_stage2_aug_comparison.py
# Stage-2 augmentation comparison runner.

import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

STAGE2_BASE_CFG = ROOT / "configs" / "base" / "stage2.yaml"
DATASET_CFG_DIR = ROOT / "configs" / "datasets"
AUG_CFG_DIR = ROOT / "configs" / "augs"
MODEL_CFG = ROOT / "configs" / "models" / "resnet18.yaml"
SELECTED_LR_CFG = ROOT / "configs" / "protocol" / "stage2_selected_lrs.yaml"


DATASET_EPOCHS = {
    "cifar10": 100,
    "cifar100": 100,
    "dermamnist": 50,
    "pathmnist": 50,
}


def load_selected_lrs():
    with open(SELECTED_LR_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_augs():
    augs = []
    for p in sorted(AUG_CFG_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        augs.append((p.stem, str(p)))
    return augs


def build_cmd(ds_name: str, aug_name: str, aug_yaml: str):
    selected_lrs = load_selected_lrs()

    if ds_name not in selected_lrs:
        raise KeyError(f"No selected LR found for dataset: {ds_name}")

    if ds_name not in DATASET_EPOCHS:
        raise KeyError(f"No epoch setting found for dataset: {ds_name}")

    ds_yaml = DATASET_CFG_DIR / f"{ds_name}.yaml"
    if not ds_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {ds_yaml}")

    lr = selected_lrs[ds_name]
    epochs = DATASET_EPOCHS[ds_name]

    return [
        "python",
        str(ROOT / "main.py"),
        str(STAGE2_BASE_CFG),
        str(ds_yaml),
        str(MODEL_CFG),
        str(aug_yaml),
        "--override",
        "data.use_val_split=false",
        "early_stopping.enabled=false",
        "wandb.project=stage2-aug-comparison-v4",
        f"wandb.group={ds_name}",
        f"train.lr={lr}",
        f"train.epochs={epochs}",
        f"train.exp_id=S2_{aug_name}",
    ]


def run_one(ds_name: str, aug_name: str, aug_yaml: str, idx: int, total: int):
    print("=" * 80)
    print(f"[{idx}/{total}] [Stage-2] dataset={ds_name}, aug={aug_name}")
    cmd = build_cmd(ds_name, aug_name, aug_yaml)
    print("Command:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python runners/run_stage2_aug_comparison.py <dataset> [aug_name]")
        sys.exit(1)

    ds_name = sys.argv[1].lower()
    allowed = ["cifar10", "cifar100", "dermamnist", "pathmnist"]
    if ds_name not in allowed:
        print(f"[ERROR] Unknown dataset: {ds_name}")
        sys.exit(1)

    requested_aug = sys.argv[2].lower() if len(sys.argv) > 2 else None

    all_augs = discover_augs()
    if not all_augs:
        print("[ERROR] No augmentation yaml files found.")
        sys.exit(1)

    all_aug_names = [name for name, _ in all_augs]
    print("Available augmentations:", ", ".join(all_aug_names))

    if requested_aug is not None:
        all_augs = [(name, y) for name, y in all_augs if name == requested_aug]
        if not all_augs:
            print(f"[ERROR] Augmentation '{requested_aug}' not found.")
            sys.exit(1)
        print(f"Will run only augmentation: {requested_aug}")

    total = len(all_augs)
    print(f"\nWill run Stage-2 for dataset '{ds_name}' with {total} augmentation(s).\n")

    for idx, (aug_name, aug_yaml) in enumerate(all_augs, start=1):
        run_one(ds_name, aug_name, aug_yaml, idx, total)

    print("\nAll Stage-2 runs finished.\n")


if __name__ == "__main__":
    main()