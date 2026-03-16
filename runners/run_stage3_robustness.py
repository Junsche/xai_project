# runners/run_stage3_robustness.py
# -*- coding: utf-8 -*-

"""
Stage-3 runner wrapper.

EN:
- Supports model switching via --model
- If aug is omitted, runs all default augmentations
- If severity is omitted, eval/stage3.py decides the default behavior

ZH:
- 通过 --model 支持模型切换
- 如果省略 aug，则自动运行默认 augmentation 列表
- 如果省略 severity，则由 eval/stage3.py 决定默认行为
"""

import argparse
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUGS_CFG = ROOT / "configs" / "protocol" / "default_augmentations.yaml"

MODEL_CFG_MAP = {
    "resnet18": ROOT / "configs" / "models" / "resnet18.yaml",
    "vit_b": ROOT / "configs" / "models" / "vit_b.yaml",
}


def load_default_augs():
    with open(DEFAULT_AUGS_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    augs = cfg.get("augmentations", [])
    if not augs:
        raise ValueError(f"No augmentations found in {DEFAULT_AUGS_CFG}")
    return augs


def run_one(dataset: str, model: str, aug: str, severity: str | None = None):
    model_cfg = MODEL_CFG_MAP[model]

    cmd = [
        "python",
        str(ROOT / "eval" / "stage3.py"),
        dataset,
        aug,
        "--model-cfg",
        str(model_cfg),
    ]

    if severity is not None:
        cmd.extend(["--severity", str(severity)])

    print("=" * 100)
    print(f"[Stage-3] model={model}, dataset={dataset}, aug={aug}, severity={severity}")
    print("Command:", " ".join(cmd))
    print("=" * 100)

    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage-3 robustness evaluation through the wrapper."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name: cifar10 / cifar100 / dermamnist / pathmnist",
    )
    parser.add_argument(
        "aug",
        nargs="?",
        default=None,
        help="Optional augmentation name. If omitted, run all default augmentations.",
    )
    parser.add_argument(
        "severity",
        nargs="?",
        default=None,
        help="Optional severity level. If omitted, use eval/stage3.py default behavior.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "vit_b"],
        help="Model architecture to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset.lower().strip()
    aug = args.aug.lower().strip() if args.aug is not None else None
    severity = args.severity
    model = args.model.lower().strip()

    allowed_datasets = ["cifar10", "cifar100", "dermamnist", "pathmnist"]
    if dataset not in allowed_datasets:
        raise ValueError(f"[ERROR] Unknown dataset: {dataset}")

    if model not in MODEL_CFG_MAP:
        raise ValueError(f"[ERROR] Unknown model: {model}")

    if aug is None:
        default_augs = load_default_augs()
        for a in default_augs:
            run_one(dataset, model, a, severity)
    else:
        run_one(dataset, model, aug, severity)


if __name__ == "__main__":
    main()