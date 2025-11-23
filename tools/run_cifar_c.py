# -*- coding: utf-8 -*-
"""
Stage-3: CIFAR-C Robustness Evaluation
阶段 3：在 CIFAR-C 上进行鲁棒性评估

EN:
    This script evaluates a trained Stage-2 model on CIFAR-10-C / CIFAR-100-C.
    For each corruption (19 types) and severity level (1–5), it loads the
    Stage-2 checkpoint, runs inference, and logs the results to Weights & Biases.

    W&B structure:
        project = "robustness-c"
        group   = "<dataset>_<augmentation>"
        name    = "resnet18_<dataset>_<aug>_<corruption>_s<severity>"
        tags    = [dataset, aug, corruption, severity]

    Usage:
        # Mode A: run ALL augmentations for a dataset
        python tools/run_cifar_c.py cifar10
        python tools/run_cifar_c.py cifar100

        # Mode B: run only ONE augmentation
        python tools/run_cifar_c.py cifar10 baseline
        python tools/run_cifar_c.py cifar10 randaugment --severity 3

ZH:
    本脚本用于在 CIFAR-10-C / CIFAR-100-C 上评估 Stage-2 训练得到的模型。
    对每一种 corruption（19 类）和每个 severity（1–5），加载 Stage-2 最优权重，
    执行推理并将结果记录到 Weights & Biases。

    W&B 日志结构：
        project = "robustness-c"
        group   = "<dataset>_<augmentation>"
        name    = "resnet18_<dataset>_<aug>_<corruption>_s<severity>"
        tags    = [dataset, aug, corruption, severity]

    用法：
        # 模式 A：对某个数据集一次性评估所有增强方法
        python tools/run_cifar_c.py cifar10
        python tools/run_cifar_c.py cifar100

        # 模式 B：只评估单个增强（可选指定 severity）
        python tools/run_cifar_c.py cifar10 baseline
        python tools/run_cifar_c.py cifar10 randaugment --severity 3
"""

import os
import sys
import argparse
from pathlib import Path

# ------------------------------------------------------------
# 修复 Python import 路径问题：把项目根目录加入 sys.path
# Fix Python import path: add project root to sys.path
# ------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import wandb

from models.factory import build_model
from train.trainer import evaluate
from xai_data.cifar_c import make_cifar_c_loaders
from utils.seed import seed_everything


# ============================================================
# 1. Dataset info + LR from Stage-1
# ============================================================

DATASETS = {
    "cifar10": {
        "num_classes": 10,
        "c_root": "./data/CIFAR-10-C",
        "lr": 0.01,
    },
    "cifar100": {
        "num_classes": 100,
        "c_root": "./data/CIFAR-100-C",
        "lr": 0.05,
    },
}

# 默认在 Stage-3 中评估的 augmentation 列表
# Default list of augmentations to evaluate in Stage-3
DEFAULT_AUGS = [
    "baseline",
    "autoaugment",
    "randaugment",
    "rotation_erasing",
    "mixup",
    "cutmix",
    "styleaug",
    "diffusemix",
]

MODEL_NAME = "resnet18"
SEED = 1437
CKPT_DIR = "./runs"


# ============================================================
# 2. Build checkpoint filename (根据 Stage-2 命名规则)
# ============================================================

def build_ckpt_path(dataset: str, aug: str) -> str:
    """
    Stage-2 checkpoint name pattern:
        resnet18_cifar10_styleaug_S2_styleaug_lr0.01_seed1437_best.pt
    """
    lr = DATASETS[dataset]["lr"]
    exp_id = f"S2_{aug}"

    run_name = (
        f"{MODEL_NAME}_{dataset}_{aug}_"
        f"{exp_id}_lr{lr}_seed{SEED}"
    )

    return os.path.join(CKPT_DIR, f"{run_name}_best.pt")


# ============================================================
# 3. Evaluate for all corruptions × severities
# ============================================================

def eval_on_cifar_c(dataset: str, aug: str, severity: int | None):
    assert dataset in DATASETS, f"Unknown dataset: {dataset}"

    num_classes = DATASETS[dataset]["num_classes"]
    c_root = DATASETS[dataset]["c_root"]

    # 1) Seed + device
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Load Stage-2 checkpoint
    ckpt_path = build_ckpt_path(dataset, aug)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model = build_model(MODEL_NAME, num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3) Build CIFAR-C loaders
    print(f"[INFO] Loading CIFAR-C from: {c_root}")
    loaders = make_cifar_c_loaders(root=c_root, batch_size=128, num_workers=4)

    all_results = {}

    # 4) Loop corruption × severity
    for cname, sev_loaders in loaders.items():
        for sev_idx, loader in enumerate(sev_loaders, start=1):

            # 如果指定 severity，则仅评估一个
            # If severity is specified, only evaluate that one
            if severity is not None and sev_idx != severity:
                continue

            print("=" * 80)
            print(f"[EVAL] dataset={dataset}, aug={aug}, corruption={cname}, severity={sev_idx}")

            # -----------------------------
            # W&B run name / group / tags
            # -----------------------------
            run_name = f"{MODEL_NAME}_{dataset}_{aug}_{cname}_s{sev_idx}"

            wandb.init(
                project="robustness-c",
                group=f"{dataset}_{aug}",   # e.g., cifar10_styleaug
                name=run_name,
                tags=[dataset, aug, cname, f"severity{sev_idx}"],
                config={
                    "dataset": dataset,
                    "augmentation": aug,
                    "corruption": cname,
                    "severity": sev_idx,
                    "checkpoint": ckpt_path,
                },
                settings=wandb.Settings(code_dir="."),
            )

            # -----------------------------
            # Evaluate on this corruption + severity
            # -----------------------------
            acc, ece, _ = evaluate(
                model,
                loader,
                device,
                do_ece=True,
                do_bal_acc=False,
            )

            print(f"[RESULT] acc={acc:.4f}, ece={ece:.4f}")

            wandb.log({
                "eval/acc": acc,
                "eval/ece": ece,
                "meta/corruption": cname,
                "meta/severity": sev_idx,
            })

            wandb.finish()

            # Local storage
            all_results[(cname, sev_idx)] = {"acc": acc, "ece": ece}

    # 5) Print summary
    print(f"\n================ SUMMARY ({dataset}, {aug}) ================")
    for (c, s), m in sorted(all_results.items()):
        print(f"{c:20s}  s={s}:  acc={m['acc']:.4f}, ece={m['ece']:.4f}")
    print("============================================================\n")


# ============================================================
# 4. CLI interface
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # 第一个位置参数：数据集
    # First positional arg: dataset
    parser.add_argument(
        "dataset",
        choices=["cifar10", "cifar100"],
        help="Dataset name (cifar10 / cifar100)",
    )

    # 第二个位置参数：augmentation（可以省略）
    # Second positional arg: augmentation (optional)
    parser.add_argument(
        "aug",
        nargs="?",           # <- makes it optional
        help="Augmentation name (baseline / randaugment / mixup / cutmix / styleaug / ...). "
             "If omitted, run ALL default augmentations.",
    )

    parser.add_argument(
        "--severity",
        type=int,
        default=None,
        help="If set (1–5), only evaluate that severity; otherwise evaluate all severities.",
    )

    args = parser.parse_args()

    if args.severity is not None and not (1 <= args.severity <= 5):
        raise ValueError("--severity must be between 1 and 5")

    dataset = args.dataset

    # ---------------- Mode A: run ALL augmentations ----------------
    if args.aug is None:
        print(f"[INFO] No augmentation specified, will run ALL default augs for {dataset}:")
        print("       " + ", ".join(DEFAULT_AUGS))
        for aug in DEFAULT_AUGS:
            print("\n" + "#" * 80)
            print(f"[STAGE-3] dataset={dataset}, aug={aug}")
            print("#" * 80)
            eval_on_cifar_c(dataset, aug, args.severity)

    # ---------------- Mode B: run ONE augmentation ----------------
    else:
        aug = args.aug
        # 简单检查：如果不在 DEFAULT_AUGS，给出提示但仍允许运行
        if aug not in DEFAULT_AUGS:
            print(f"[WARN] aug='{aug}' is not in DEFAULT_AUGS list "
                  f"({', '.join(DEFAULT_AUGS)}). Make sure Stage-2 checkpoint name matches.")
        eval_on_cifar_c(dataset, aug, args.severity)


if __name__ == "__main__":
    main()