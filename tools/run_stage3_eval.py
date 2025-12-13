# -*- coding: utf-8 -*-
"""
Stage-3: Clean Test + CIFAR-C Robustness Evaluation
阶段 3：在干净测试集和 CIFAR-C 上评估 Stage-2 模型的鲁棒性

EN:
    This script loads Stage-2 checkpoints (last.pt) for a given
    dataset + augmentation and evaluates them on:
        1) the clean CIFAR test set (no corruptions)
        2) the CIFAR-C corrupted test sets (19 corruptions × 5 severities)

    Results are logged to Weights & Biases with a consistent structure:
        - project:  "robustness-stage3"
        - group:    "<dataset>_<augmentation>"
        - metrics:  eval/acc, eval/ece, eval/mce, eval/loss
        - metadata: meta/corruption, meta/severity

    Usage examples:
        # evaluate ALL default augmentations for CIFAR-10
        python tools/run_stage3_eval.py cifar10

        # evaluate ALL default augmentations for CIFAR-100
        python tools/run_stage3_eval.py cifar100

        # evaluate ONLY 'baseline' on CIFAR-10
        python tools/run_stage3_eval.py cifar10 baseline

        # evaluate ONLY 'randaugment' on CIFAR-10 with CIFAR-C severity = 3
        python tools/run_stage3_eval.py cifar10 randaugment --severity 3


ZH:
    本脚本用于在「干净 CIFAR 测试集」和「CIFAR-C」上评估 Stage-2 训练好的模型
    （即 *_last.pt checkpoint）。

    对于指定的数据集 + 增强方式，本脚本会：
        1) 在不带任何腐蚀的 CIFAR 测试集上评估一次（clean test）
        2) 在 CIFAR-C 上对 19 种 corruption × 5 个 severity 做推理评估

    所有结果统一记录到 W&B，结构为：
        - project:  "robustness-stage3"
        - group:    "<dataset>_<augmentation>"
        - 指标:     eval/acc, eval/ece, eval/mce, eval/loss
        - 元信息:   meta/corruption, meta/severity

    使用示例：
        # 一次性评估 CIFAR-10 上所有默认增强
        python tools/run_stage3_eval.py cifar10

        # 一次性评估 CIFAR-100 上所有默认增强
        python tools/run_stage3_eval.py cifar100

        # 只评估 CIFAR-10 上的 baseline
        python tools/run_stage3_eval.py cifar10 baseline

        # 只评估 CIFAR-10 上 randaugment 在 severity = 3 的 CIFAR-C 表现
        python tools/run_stage3_eval.py cifar10 randaugment --severity 3
"""

import os
import sys
import argparse
from typing import Optional

import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets

# ---------------------------------------------------------------------
# Make sure we can import from the project root
# 确保可以从项目根目录导入模块
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factory import build_model
from train.trainer import evaluate
from xai_data.cifar_c import make_cifar_c_loaders
from xai_data.transforms_registry import REGISTRY
from utils.seed import seed_everything


# ============================================================
# Load dataset config from YAML
# 从 YAML 加载数据集配置（root、c_root、num_classes）
# ============================================================

def _load_dataset_cfg(name: str) -> dict:
    """
    EN:
        Load dataset-specific config from configs/datasets/<name>.yaml.
        Used to obtain:
            - data.root
            - data.cifar_c_root
            - model.num_classes

    ZH：
        从 configs/datasets/<name>.yaml 中加载数据集配置，
        用于获取：
            - data.root
            - data.cifar_c_root
            - model.num_classes
    """
    cfg_path = os.path.join(ROOT, "configs", "datasets", f"{name}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


_cfg_c10 = _load_dataset_cfg("cifar10")
_cfg_c100 = _load_dataset_cfg("cifar100")

DATASETS = {
    "cifar10": {
        "num_classes": _cfg_c10["model"]["num_classes"],
        "root": _cfg_c10["data"]["root"],              # e.g. "./data/CIFAR-10"
        "c_root": _cfg_c10["data"]["cifar_c_root"],    # e.g. "./data/CIFAR-10-C"
    },
    "cifar100": {
        "num_classes": _cfg_c100["model"]["num_classes"],
        "root": _cfg_c100["data"]["root"],             # e.g. "./data/CIFAR-100"
        "c_root": _cfg_c100["data"]["cifar_c_root"],   # e.g. "./data/CIFAR-100-C"
    },
}

# 默认需要评估的增强方法（需与 configs/augs/*.yaml 中的名字一致）
# Default augmentations to evaluate (must match configs/augs/*.yaml / data.aug)
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

WANDB_PROJECT = "robustness-stage3"


# ============================================================
# Helper: build Stage-2 checkpoint path
#         构造 Stage-2 checkpoint 路径（包含 mixup/cutmix 修正）
# ============================================================

def build_ckpt_path(dataset: str, aug: str) -> str:
    """
    EN:
        Build correct Stage-2 checkpoint filename.

        Special note:
        - mixup and cutmix use data.aug = baseline in Stage-2,
          so the actual checkpoint name includes "baseline"
          instead of "mixup" / "cutmix" in the third token.

        Example filenames (CIFAR-10):
            resnet18_cifar10_baseline_S2_mixup_lr0.01_seed1437_last.pt
            resnet18_cifar10_baseline_S2_cutmix_lr0.01_seed1437_last.pt
            resnet18_cifar10_randaugment_S2_randaugment_lr0.01_seed1437_last.pt

    ZH：
        构造正确的 Stage-2 checkpoint 路径。

        注意：
        - mixup 和 cutmix 在 Stage-2 中的 data.aug 均为 baseline，
          所以 checkpoint 名称的第三段应该是 baseline，
          而不是 mixup / cutmix。

        例如（CIFAR-10）：
            resnet18_cifar10_baseline_S2_mixup_lr0.01_seed1437_last.pt
            resnet18_cifar10_baseline_S2_cutmix_lr0.01_seed1437_last.pt
            resnet18_cifar10_randaugment_S2_randaugment_lr0.01_seed1437_last.pt
    """
    dataset = dataset.lower()

    # learning rate used in Stage-2
    if dataset == "cifar10":
        lr = "0.01"
    elif dataset == "cifar100":
        lr = "0.05"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Stage-2 real naming rule:
    # resnet18_<dataset>_<aug_token>_S2_<aug>_lr<lr>_seed1437_last.pt
    if aug in ["mixup", "cutmix"]:
        aug_token = "baseline"   # 关键修正：文件名第三段仍是 baseline
    else:
        aug_token = aug

    exp_id = f"S2_{aug}"
    run_name = f"{MODEL_NAME}_{dataset}_{aug_token}_{exp_id}_lr{lr}_seed{SEED}"

    return os.path.join(CKPT_DIR, f"{run_name}_last.pt")


# ============================================================
# Helper: clean CIFAR test loader
#         构造干净 CIFAR 测试集的 DataLoader
# ============================================================

def make_clean_test_loader(dataset: str,
                           batch_size: int = 128,
                           num_workers: int = 4,
                           root: Optional[str] = None) -> DataLoader:
    """
    EN:
        Build a clean CIFAR-10 / CIFAR-100 test loader using the SAME root
        directory as Stage-1 / Stage-2 (cfg["data"]["root"]).

    ZH：
        构建干净 CIFAR 测试集的 DataLoader，使用与 Stage-1 / Stage-2
        相同的 root 路径（即 cfg["data"]["root"]），而不是固定的 ./data。
    """
    dataset = dataset.lower()
    if dataset == "cifar10":
        ds_cls = datasets.CIFAR10
    elif dataset == "cifar100":
        ds_cls = datasets.CIFAR100
    else:
        raise ValueError(f"Clean test loader only supports CIFAR, got {dataset}")

    if root is None:
        raise ValueError("Clean test loader requires a dataset root path!")

    # baseline augment 的 test transform（通常只有 ToTensor + Normalize）
    # Use 'baseline' test transform to keep normalization consistent.
    _, test_tf = REGISTRY["baseline"]()

    test_ds = ds_cls(root=root, train=False, download=False, transform=test_tf)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ============================================================
# Evaluation on clean test
# 在干净测试集上的评估
# ============================================================

def eval_clean_test(dataset: str, aug: str) -> None:
    """
    EN:
        Evaluate Stage-2 checkpoint on the *clean* CIFAR test set
        (no corruptions). Logs:
            - eval/acc, eval/ece, eval/mce, eval/loss
            - meta/corruption = "clean"
            - meta/severity   = 0

    ZH：
        在「干净 CIFAR 测试集」上评估 Stage-2 checkpoint（无任何 corruption）。
        记录的指标包括：
            - eval/acc, eval/ece, eval/mce, eval/loss
            - meta/corruption = "clean"
            - meta/severity   = 0
    """
    dataset = dataset.lower()
    assert dataset in DATASETS, f"Unknown dataset: {dataset}"

    num_classes = DATASETS[dataset]["num_classes"]
    root_path = DATASETS[dataset]["root"]

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = build_ckpt_path(dataset, aug)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[CLEAN] Checkpoint not found: {ckpt_path}")

    print(f"[CLEAN] Loading checkpoint: {ckpt_path}")

    model = build_model(MODEL_NAME, num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_loader = make_clean_test_loader(
        dataset,
        batch_size=128,
        num_workers=4,
        root=root_path,
    )

    run_name = f"{MODEL_NAME}_{dataset}_{aug}_clean_test"

    wandb.init(
        project=WANDB_PROJECT,
        group=f"{dataset}_{aug}",
        name=run_name,
        tags=[dataset, aug, "clean"],
        config={
            "stage": "stage3",
            "dataset": dataset,
            "augmentation": aug,
            "corruption": "clean",
            "severity": 0,
            "checkpoint": ckpt_path,
        },
        settings=wandb.Settings(code_dir="."),
    )

    acc, ece, mce, _, avg_loss = evaluate(
        model,
        test_loader,
        device,
        do_ece=True,
        do_bal_acc=False,
    )

    print(f"[CLEAN RESULT] acc={acc:.4f}, ece={ece:.4f}, mce={mce:.4f}, loss={avg_loss:.4f}")

    wandb.log({
        "eval/acc": acc,
        "eval/ece": ece,
        "eval/mce": mce,
        "eval/loss": avg_loss,
        "meta/corruption": "clean",
        "meta/severity": 0,
    })

    wandb.finish()


# ============================================================
# Evaluation on CIFAR-C
# 在 CIFAR-C 上的评估
# ============================================================

def eval_on_cifar_c(dataset: str, aug: str, severity: Optional[int]) -> None:
    """
    EN:
        Evaluate Stage-2 checkpoint on CIFAR-C (all corruptions × severities).
        For each (corruption, severity), this function:
            - runs evaluate(...)
            - logs metrics & metadata to W&B

    ZH：
        在 CIFAR-C 上评估 Stage-2 checkpoint（覆盖所有 corruption × severity）。
        对于每个 (corruption, severity)，本函数会：
            - 调用 evaluate(...)
            - 将指标和元信息写入 W&B
    """
    dataset = dataset.lower()
    assert dataset in DATASETS, f"Unknown dataset: {dataset}"

    if severity is not None and not (1 <= severity <= 5):
        raise ValueError("--severity must be between 1 and 5")

    num_classes = DATASETS[dataset]["num_classes"]
    c_root = DATASETS[dataset]["c_root"]

    # 评估 CIFAR-C 时也需要固定随机种子
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = build_ckpt_path(dataset, aug)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[CIFAR-C] Checkpoint not found: {ckpt_path}")

    print(f"[CIFAR-C] Loading checkpoint: {ckpt_path}")

    model = build_model(MODEL_NAME, num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loaders = make_cifar_c_loaders(root=c_root, batch_size=128, num_workers=4)

    all_results = {}

    for cname, sev_loaders in loaders.items():
        for sev_idx, loader in enumerate(sev_loaders, start=1):

            if severity is not None and severity != sev_idx:
                continue

            print("=" * 80)
            print(f"[CIFAR-C EVAL] dataset={dataset}, aug={aug}, "
                  f"corruption={cname}, severity={sev_idx}")

            run_name = f"{MODEL_NAME}_{dataset}_{aug}_{cname}_s{sev_idx}"

            wandb.init(
                project=WANDB_PROJECT,
                group=f"{dataset}_{aug}",
                name=run_name,
                tags=[dataset, aug, cname, f"severity{sev_idx}", "cifar-c"],
                config={
                    "stage": "stage3",
                    "dataset": dataset,
                    "augmentation": aug,
                    "corruption": cname,
                    "severity": sev_idx,
                    "checkpoint": ckpt_path,
                },
                settings=wandb.Settings(code_dir="."),
            )

            acc, ece, mce, _, avg_loss = evaluate(
                model,
                loader,
                device,
                do_ece=True,
                do_bal_acc=False,
            )

            print(f"[CIFAR-C RESULT] corruption={cname:20s} "
                  f"s={sev_idx}: acc={acc:.4f}, ece={ece:.4f}, "
                  f"mce={mce:.4f}, loss={avg_loss:.4f}")

            wandb.log({
                "eval/acc": acc,
                "eval/ece": ece,
                "eval/mce": mce,
                "eval/loss": avg_loss,
                "meta/corruption": cname,
                "meta/severity": sev_idx,
            })

            wandb.finish()

            all_results[(cname, sev_idx)] = {
                "acc": acc,
                "ece": ece,
                "mce": mce,
                "loss": avg_loss,
            }

    print(f"\n================ CIFAR-C SUMMARY ({dataset}, {aug}) ================")
    for (c, s), m in sorted(all_results.items()):
        print(
            f"{c:20s}  s={s}:  acc={m['acc']:.4f}, "
            f"ece={m['ece']:.4f}, mce={m['mce']:.4f}, loss={m['loss']:.4f}"
        )
    print("=====================================================================\n")


# ============================================================
# CLI entry point
# 命令行入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-3 evaluation on clean test + CIFAR-C"
    )

    parser.add_argument(
        "dataset",
        choices=["cifar10", "cifar100"],
        help="Dataset name (cifar10 / cifar100)"
    )

    parser.add_argument(
        "aug",
        nargs="?",
        help=(
            "Augmentation name (optional). If omitted, all DEFAULT_AUGS will "
            "be evaluated."
        ),
    )

    parser.add_argument(
        "--severity",
        type=int,
        default=None,
        help="If set (1–5), evaluate only that CIFAR-C severity level.",
    )

    args = parser.parse_args()

    dataset = args.dataset.lower()
    aug = args.aug.lower() if args.aug is not None else None

    # mode A: run ALL default augmentations for the dataset
    # 模式 A：对该数据集的所有默认增强做评估
    if aug is None:
        print(f"[INFO] Running Stage-3 for ALL augmentations on {dataset}")
        print(f"       DEFAULT_AUGS = {', '.join(DEFAULT_AUGS)}\n")

        for a in DEFAULT_AUGS:
            print("\n" + "#" * 80)
            print(f"[STAGE-3] dataset={dataset}, aug={a} (clean + CIFAR-C)")
            print("#" * 80)

            # 先跑干净测试集
            eval_clean_test(dataset, a)
            # 再跑 CIFAR-C
            eval_on_cifar_c(dataset, a, args.severity)

    # mode B: run only ONE augmentation
    # 模式 B：只评估某一个增强
    else:
        if aug not in DEFAULT_AUGS:
            print(f"[WARN] Aug '{aug}' not in DEFAULT_AUGS, "
                  f"but will still be evaluated.")
        print(f"[STAGE-3] dataset={dataset}, aug={aug} (clean + CIFAR-C)")
        eval_clean_test(dataset, aug)
        eval_on_cifar_c(dataset, aug, args.severity)


if __name__ == "__main__":
    main()