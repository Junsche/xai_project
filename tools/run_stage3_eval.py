# -*- coding: utf-8 -*-
"""
Stage-3: Clean Test + CIFAR-C Robustness Evaluation
阶段 3：在干净测试集和 CIFAR-C 上评估 Stage-2 模型的鲁棒性

EN:
    - Loads Stage-2 checkpoints (*_last.pt)
    - Evaluates on:
        1) clean CIFAR test set
        2) CIFAR-C (19 corruptions x 5 severities)
    - Logs to W&B:
        eval/acc, eval/ece, eval/mce, eval/loss
        meta/corruption, meta/severity

    IMPORTANT:
    - Stage-3 should NOT download pretrained weights.
      We force model_cfg["weights"]="none" because we load checkpoint anyway.

ZH:
    - 加载 Stage-2 的 *_last.pt
    - 在 clean test + CIFAR-C 上评估
    - 写入 W&B 指标与元信息
    - 重要：Stage-3 不需要 pretrained，会强制 weights=none，避免下载
"""

import os
import sys
import argparse
from typing import Optional, Dict

import yaml
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets

# ---------------------------------------------------------------------
# Ensure imports from project root
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factory import build_model
from train.trainer import evaluate
from xai_data.cifar_c import make_cifar_c_loaders
from xai_data.transforms_registry import REGISTRY
from utils.seed import seed_everything


# -------------------------
# Defaults
# -------------------------
DEFAULT_AUGS = [
    "baseline",
    "autoaugment",
    "randaugment",
    "augmix",
    "rotation_erasing",
    "mixup",
    "cutmix",
    "styleaug",
    "diffusemix",
]

WANDB_PROJECT = "robustness-stage3"
SEED = 1437
CKPT_DIR = "./runs"


# ============================================================
# YAML helpers
# ============================================================

def load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_cfg(dataset: str) -> dict:
    path = os.path.join(ROOT, "configs", "datasets", f"{dataset}.yaml")
    return load_yaml(path)


def get_dataset_info(dataset: str) -> Dict[str, str]:
    cfg = load_dataset_cfg(dataset)
    return {
        "num_classes": cfg["model"]["num_classes"],
        "root": cfg["data"]["root"],
        "c_root": cfg["data"]["cifar_c_root"],
    }


def load_model_cfg(base_path: str) -> dict:
    base = load_yaml(base_path)
    model_cfg = base.get("model", {})
    if "name" not in model_cfg:
        raise KeyError(f"`model.name` missing in {base_path}")
    model_cfg = dict(model_cfg)
    model_cfg.setdefault("weights", "none")
    return model_cfg


# ============================================================
# Checkpoint path
# ============================================================

def stage2_lr_for(dataset: str) -> str:
    if dataset == "cifar10":
        return "0.01"
    if dataset == "cifar100":
        return "0.05"
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_ckpt_path(dataset: str, aug: str, model_name: str) -> str:
    """
    Must match Stage-2 naming:
    <model>_<dataset>_<aug_token>_S2_<aug>_lr<lr>_seed<seed>_last.pt

    mixup/cutmix special case:
      checkpoint name uses aug_token="baseline"
    """
    lr = stage2_lr_for(dataset)

    if aug in ["mixup", "cutmix"]:
        aug_token = "baseline"
    else:
        aug_token = aug

    exp_id = f"S2_{aug}"
    run_name = f"{model_name}_{dataset}_{aug_token}_{exp_id}_lr{lr}_seed{SEED}"
    return os.path.join(CKPT_DIR, f"{run_name}_last.pt")


# ============================================================
# Data loaders
# ============================================================

def make_clean_test_loader(dataset: str, root: str, batch_size: int = 128, num_workers: int = 4) -> DataLoader:
    if dataset == "cifar10":
        ds_cls = datasets.CIFAR10
    elif dataset == "cifar100":
        ds_cls = datasets.CIFAR100
    else:
        raise ValueError(f"Clean test loader supports cifar10/100, got {dataset}")

    _, test_tf = REGISTRY["baseline"]()
    ds = ds_cls(root=root, train=False, download=False, transform=test_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# ============================================================
# Model loading
# ============================================================

def build_model_for_stage3(model_cfg: dict, num_classes: int, device: torch.device):
    """
    EN: Force weights='none' to avoid downloading. We'll load checkpoint anyway.
    ZH: 强制 weights='none' 避免下载。反正马上 load checkpoint。
    """
    cfg = dict(model_cfg)
    cfg["weights"] = "none"
    model = build_model(cfg, num_classes).to(device)
    return model


def load_ckpt(model, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()


# ============================================================
# Evaluation
# ============================================================

def eval_clean(dataset: str, aug: str, model_cfg: dict):
    info = get_dataset_info(dataset)
    num_classes = info["num_classes"]
    root_path = info["root"]

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_cfg["name"]
    ckpt_path = build_ckpt_path(dataset, aug, model_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[CLEAN] Checkpoint not found: {ckpt_path}")

    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    loader = make_clean_test_loader(dataset, root_path)

    wandb.init(
        project=WANDB_PROJECT,
        group=f"{dataset}_{aug}",
        name=f"{model_name}_{dataset}_{aug}_clean",
        tags=[dataset, aug, "clean"],
        config={
            "stage": "stage3",
            "dataset": dataset,
            "augmentation": aug,
            "corruption": "clean",
            "severity": 0,
            "checkpoint": ckpt_path,
            "model": {"name": model_cfg["name"], "weights": "none"},
        },
        settings=wandb.Settings(code_dir="."),
    )

    acc, ece, mce, _, loss = evaluate(model, loader, device, do_ece=True, do_bal_acc=False)

    wandb.log({
        "eval/acc": acc,
        "eval/ece": ece,
        "eval/mce": mce,
        "eval/loss": loss,
        "meta/corruption": "clean",
        "meta/severity": 0,
    })
    wandb.finish()


def eval_cifar_c(dataset: str, aug: str, model_cfg: dict, severity: Optional[int]):
    if severity is not None and not (1 <= severity <= 5):
        raise ValueError("--severity must be 1..5")

    info = get_dataset_info(dataset)
    num_classes = info["num_classes"]
    c_root = info["c_root"]

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_cfg["name"]
    ckpt_path = build_ckpt_path(dataset, aug, model_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[CIFAR-C] Checkpoint not found: {ckpt_path}")

    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    loaders = make_cifar_c_loaders(root=c_root, batch_size=128, num_workers=4)

    for cname, sev_loaders in loaders.items():
        for sev_idx, loader in enumerate(sev_loaders, start=1):
            if severity is not None and sev_idx != severity:
                continue

            wandb.init(
                project=WANDB_PROJECT,
                group=f"{dataset}_{aug}",
                name=f"{model_name}_{dataset}_{aug}_{cname}_s{sev_idx}",
                tags=[dataset, aug, cname, f"severity{sev_idx}", "cifar-c"],
                config={
                    "stage": "stage3",
                    "dataset": dataset,
                    "augmentation": aug,
                    "corruption": cname,
                    "severity": sev_idx,
                    "checkpoint": ckpt_path,
                    "model": {"name": model_cfg["name"], "weights": "none"},
                },
                settings=wandb.Settings(code_dir="."),
            )

            acc, ece, mce, _, loss = evaluate(model, loader, device, do_ece=True, do_bal_acc=False)

            wandb.log({
                "eval/acc": acc,
                "eval/ece": ece,
                "eval/mce": mce,
                "eval/loss": loss,
                "meta/corruption": cname,
                "meta/severity": sev_idx,
            })
            wandb.finish()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage-3 evaluation on clean test + CIFAR-C")
    parser.add_argument("dataset", choices=["cifar10", "cifar100"])
    parser.add_argument("aug", nargs="?", default=None)
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument(
        "--base",
        type=str,
        default=os.path.join(ROOT, "configs", "_base.yaml"),
        help="Base config path (default: configs/_base.yaml)",
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    aug = args.aug.lower() if args.aug is not None else None

    model_cfg = load_model_cfg(args.base)
    if model_cfg["name"].lower() != "resnet18":
        raise ValueError("This Stage-3 script currently expects resnet18 checkpoints.")

    if aug is None:
        for a in DEFAULT_AUGS:
            print(f"[STAGE-3] dataset={dataset}, aug={a}")
            eval_clean(dataset, a, model_cfg)
            eval_cifar_c(dataset, a, model_cfg, args.severity)
    else:
        print(f"[STAGE-3] dataset={dataset}, aug={aug}")
        eval_clean(dataset, aug, model_cfg)
        eval_cifar_c(dataset, aug, model_cfg, args.severity)


if __name__ == "__main__":
    main()