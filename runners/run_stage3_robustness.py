# runners/run_stage3_robustness.py
# Transition version: wrapper now calls eval/stage3.py directly.

import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUGS_CFG = ROOT / "configs" / "protocol" / "default_augmentations.yaml"
STAGE3_BASE_CFG = ROOT / "configs" / "base" / "stage3.yaml"
MODEL_CFG = ROOT / "configs" / "models" / "resnet18.yaml"


def load_default_augs():
    with open(DEFAULT_AUGS_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    augs = cfg.get("augmentations", [])
    if not augs:
        raise ValueError(f"No augmentations found in {DEFAULT_AUGS_CFG}")
    return augs


def run_one(dataset: str, aug: str, severity: str | None = None):
    cmd = [
        "python",
        str(ROOT / "eval" / "stage3.py"),
        dataset,
        aug,
        "--base",
        str(STAGE3_BASE_CFG),
        "--model-cfg",
        str(MODEL_CFG),
    ]

    if severity is not None:
        cmd.extend(["--severity", str(severity)])

    print("=" * 80)
    print(f"[Stage-3] dataset={dataset}, aug={aug}, severity={severity}")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python runners/run_stage3_robustness.py <dataset> [aug] [severity]")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    allowed = ["cifar10", "cifar100", "dermamnist", "pathmnist"]
    if dataset not in allowed:
        print(f"[ERROR] Unknown dataset: {dataset}")
        sys.exit(1)

    aug = sys.argv[2].lower() if len(sys.argv) > 2 else None
    severity = sys.argv[3] if len(sys.argv) > 3 else None

    if aug is None:
        default_augs = load_default_augs()
        for a in default_augs:
            run_one(dataset, a, severity)
    else:
        run_one(dataset, aug, severity)


if __name__ == "__main__":
    main()