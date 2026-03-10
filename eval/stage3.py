# -*- coding: utf-8 -*-
"""
Stage-3: Clean Test + Corruption Robustness Evaluation
- CIFAR10/100: clean test + CIFAR-C
- MedMNIST (DermaMNIST / PathMNIST): clean test + MedMNIST-C

Transition version:
- transitioned from the earlier Stage-3 implementation into the new eval/stage3.py entry
- adapted to the new config structure
- Stage-3 metrics are logged as test/* instead of eval/*
"""

import os
import sys
import argparse
from typing import Optional, Any, List, Tuple

import yaml
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms as tvt

# ---------------------------------------------------------------------
# Ensure imports from project root
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factory import build_model
from train.trainer import evaluate
from utils.seed import seed_everything

from data_modules.medmnist import get_loaders as get_medmnist_loaders
from data_modules.medmnist_c import make_medmnist_c_loader

from eval.checkpointing import assert_stage2_ckpt_exists

SEED = 1437


# ============================================================
# YAML helpers
# ============================================================

def load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML is empty: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"YAML must load as dict: {path}")
    return data


def load_dataset_cfg(dataset: str) -> dict:
    path = os.path.join(ROOT, "configs", "datasets", f"{dataset}.yaml")
    return load_yaml(path)


def load_stage3_cfg(path: str) -> dict:
    return load_yaml(path)


def load_model_cfg(path: str) -> dict:
    cfg = load_yaml(path)
    model_cfg = dict(cfg.get("model", {}))
    if "name" not in model_cfg:
        raise KeyError(f"`model.name` missing in {path}")
    model_cfg.setdefault("weights", "none")
    return model_cfg


def _get_list(cfg: dict, key: str) -> Optional[List[Any]]:
    v = cfg.get(key, None)
    if v is None:
        return None
    if isinstance(v, list):
        return v
    raise TypeError(f"Expected list for {key}, got {type(v)}")


def _as_tuple_floats(x, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
    if x is None:
        return fallback
    if isinstance(x, (list, tuple)):
        return tuple(float(v) for v in x)
    return (float(x),)


# ============================================================
# Model loading
# ============================================================

def build_model_for_stage3(model_cfg: dict, num_classes: int, device: torch.device):
    cfg = dict(model_cfg)
    cfg["weights"] = "none"
    model = build_model(cfg, num_classes).to(device)
    return model


def load_ckpt(model, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()


# ============================================================
# W&B helpers
# ============================================================

def wandb_run(
    *,
    project: str,
    dataset: str,
    aug: str,
    model_name: str,
    corruption: str,
    severity: int,
    ckpt_path: str,
    tags: Optional[list] = None,
    extra_config: Optional[dict] = None,
):
    cfg = {
        "stage": "stage3",
        "dataset": dataset,
        "augmentation": aug,
        "corruption": corruption,
        "severity": severity,
        "checkpoint": ckpt_path,
        "model": {"name": model_name, "weights": "none"},
    }
    if extra_config:
        cfg.update(extra_config)

    wandb.init(
        project=project,
        group=f"{dataset}_{aug}",
        name=f"{model_name}_{dataset}_{aug}_{corruption}_s{severity}",
        tags=tags or [dataset, aug, corruption, f"severity{severity}"],
        config=cfg,
        settings=wandb.Settings(code_dir="."),
    )


# ============================================================
# Data loaders: CIFAR clean
# ============================================================

def make_clean_test_loader_cifar(dataset: str, data_cfg: dict) -> DataLoader:
    if dataset == "cifar10":
        ds_cls = datasets.CIFAR10
    elif dataset == "cifar100":
        ds_cls = datasets.CIFAR100
    else:
        raise ValueError(dataset)

    img_size = int(data_cfg.get("img_size", 32))
    mean = _as_tuple_floats(data_cfg.get("mean"), (0.5, 0.5, 0.5))
    std = _as_tuple_floats(data_cfg.get("std"), (0.5, 0.5, 0.5))

    tf = tvt.Compose([
        tvt.ToTensor(),
        tvt.Resize((img_size, img_size)),
        tvt.Normalize(mean, std),
    ])

    ds = ds_cls(root=data_cfg["root"], train=False, download=False, transform=tf)
    return DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )


# ============================================================
# Data loaders: CIFAR-C
# ============================================================

def _normalize_batch_nchw(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def make_cifar_c_loaders(
    root: str,
    *,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    batch_size=128,
    num_workers=4,
):
    loaders = {}
    root = os.path.abspath(root)

    for fname in os.listdir(root):
        if not fname.endswith(".npy") or fname == "labels.npy":
            continue

        cname = fname.replace(".npy", "")
        x_all = np.load(os.path.join(root, fname))
        y_all = np.load(os.path.join(root, "labels.npy"))

        y_all = np.asarray(y_all)
        y_len = int(y_all.shape[0])

        sev_loaders = []
        for s in range(5):
            xs = x_all[s * 10000:(s + 1) * 10000]

            if y_len == 10000:
                ys = y_all
            else:
                ys = y_all[s * 10000:(s + 1) * 10000]

            if xs.ndim == 4:
                xs = xs.transpose(0, 3, 1, 2)

            xs = torch.from_numpy(xs).float() / 255.0
            ys = torch.from_numpy(np.asarray(ys)).long()

            xs = _normalize_batch_nchw(xs, mean, std)

            ds = TensorDataset(xs, ys)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            sev_loaders.append(dl)

        loaders[cname] = sev_loaders

    return loaders


# ============================================================
# Data loaders: MedMNIST clean
# ============================================================

def make_clean_test_loader_medmnist(dataset_cfg: dict) -> DataLoader:
    cfg = dict(dataset_cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["data"].setdefault("aug", "baseline")
    _, _, test_ld, _ = get_medmnist_loaders(cfg)
    return test_ld


# ============================================================
# Eval: Clean
# ============================================================

def eval_clean(dataset: str, aug: str, model_cfg: dict, stage3_cfg: dict):
    dataset_cfg = load_dataset_cfg(dataset)
    data_cfg = dataset_cfg["data"]
    eval_cfg = stage3_cfg.get("eval", {})

    do_ece = bool(eval_cfg.get("do_ece", True))
    do_bal_acc = bool(dataset_cfg.get("eval", {}).get("do_bal_acc", False))

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_cfg["name"]
    ckpt_path = assert_stage2_ckpt_exists(
        dataset=dataset,
        aug=aug,
        model_name=model_name,
        seed=SEED,
    )

    num_classes = int(dataset_cfg["model"]["num_classes"])

    if dataset in ["cifar10", "cifar100"]:
        loader = make_clean_test_loader_cifar(dataset, data_cfg)
    elif dataset in ["dermamnist", "pathmnist"]:
        loader = make_clean_test_loader_medmnist(dataset_cfg)
    else:
        raise ValueError(dataset)

    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    wandb.init(
        project=stage3_cfg["wandb"]["project"],
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
            "model": {"name": model_name, "weights": "none"},
        },
        settings=wandb.Settings(code_dir="."),
    )

    acc, ece, mce, bal_acc, loss = evaluate(
        model,
        loader,
        device,
        do_ece=do_ece,
        do_bal_acc=do_bal_acc,
    )

    wandb.log({
        "test/acc": acc,
        "test/ece": ece,
        "test/mce": mce,
        "test/loss": loss,
        "test/bal_acc": bal_acc,
        "meta/stage": "stage3",
        "meta/dataset": dataset,
        "meta/model": model_name,
        "meta/augmentation": aug,
        "meta/corruption": "clean",
        "meta/severity": 0,
    })
    wandb.finish()


# ============================================================
# Eval: CIFAR-C
# ============================================================

def eval_cifar_c(dataset: str, aug: str, model_cfg: dict, stage3_cfg: dict, severity: Optional[int]):
    dataset_cfg = load_dataset_cfg(dataset)
    data_cfg = dataset_cfg["data"]
    eval_cfg = stage3_cfg.get("eval", {})

    do_ece = bool(eval_cfg.get("do_ece", True))
    do_bal_acc = bool(dataset_cfg.get("eval", {}).get("do_bal_acc", False))

    if severity is not None and not (1 <= severity <= 5):
        raise ValueError("--severity must be 1..5")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_cfg["name"]
    ckpt_path = assert_stage2_ckpt_exists(
        dataset=dataset,
        aug=aug,
        model_name=model_name,
        seed=SEED,
    )

    num_classes = int(dataset_cfg["model"]["num_classes"])
    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    mean = _as_tuple_floats(data_cfg.get("mean"), (0.5, 0.5, 0.5))
    std = _as_tuple_floats(data_cfg.get("std"), (0.5, 0.5, 0.5))

    loaders = make_cifar_c_loaders(
        data_cfg["cifar_c_root"],
        mean=mean,
        std=std,
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 4)),
    )

    wanted_corruptions = _get_list(data_cfg, "corruptions")
    wanted_severities = _get_list(data_cfg, "severities")

    for cname, sev_loaders in loaders.items():
        if wanted_corruptions is not None and cname not in wanted_corruptions:
            continue

        for sev_idx, loader in enumerate(sev_loaders, start=1):
            if severity is not None and sev_idx != severity:
                continue
            if wanted_severities is not None and sev_idx not in wanted_severities:
                continue

            wandb_run(
                project=stage3_cfg["wandb"]["project"],
                dataset=dataset,
                aug=aug,
                model_name=model_name,
                corruption=cname,
                severity=sev_idx,
                ckpt_path=ckpt_path,
                tags=[dataset, aug, cname, f"severity{sev_idx}", "cifar-c"],
            )

            acc, ece, mce, bal_acc, loss = evaluate(
                model,
                loader,
                device,
                do_ece=do_ece,
                do_bal_acc=do_bal_acc,
            )

            wandb.log({
                "test/acc": acc,
                "test/ece": ece,
                "test/mce": mce,
                "test/loss": loss,
                "test/bal_acc": bal_acc,
                "meta/stage": "stage3",
                "meta/dataset": dataset,
                "meta/model": model_name,
                "meta/augmentation": aug,
                "meta/corruption": cname,
                "meta/severity": sev_idx,
            })
            wandb.finish()


# ============================================================
# Eval: MedMNIST-C
# ============================================================

def eval_medmnist_c(dataset: str, aug: str, model_cfg: dict, stage3_cfg: dict, severity: Optional[int]):
    dataset_cfg = load_dataset_cfg(dataset)
    data_cfg = dataset_cfg["data"]
    eval_cfg = stage3_cfg.get("eval", {})

    do_ece = bool(eval_cfg.get("do_ece", True))
    do_bal_acc = bool(dataset_cfg.get("eval", {}).get("do_bal_acc", True))

    if severity is not None and not (1 <= severity <= 5):
        raise ValueError("--severity must be 1..5")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = model_cfg["name"]
    ckpt_path = assert_stage2_ckpt_exists(
        dataset=dataset,
        aug=aug,
        model_name=model_name,
        seed=SEED,
    )

    num_classes = int(dataset_cfg["model"]["num_classes"])
    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    c_root = data_cfg["medmnist_c_root"]
    img_size = int(data_cfg.get("img_size", 28))
    mean = _as_tuple_floats(data_cfg.get("mean"), (0.5, 0.5, 0.5))
    std = _as_tuple_floats(data_cfg.get("std"), (0.5, 0.5, 0.5))

    corruptions = data_cfg.get("corruptions", None)
    if corruptions is None:
        raise KeyError(f"{dataset}.yaml missing data.corruptions for MedMNIST-C.")

    severities = data_cfg.get("severities", [1, 2, 3, 4, 5])

    for cname in corruptions:
        for sev_idx in severities:
            if severity is not None and sev_idx != severity:
                continue

            loader = make_medmnist_c_loader(
                c_root=c_root,
                dataset=dataset,
                corruption=cname,
                severity=sev_idx,
                batch_size=int(data_cfg.get("batch_size", 128)),
                num_workers=int(data_cfg.get("num_workers", 4)),
                img_size=img_size,
                mean=mean,
                std=std,
            )

            wandb_run(
                project=stage3_cfg["wandb"]["project"],
                dataset=dataset,
                aug=aug,
                model_name=model_name,
                corruption=cname,
                severity=sev_idx,
                ckpt_path=ckpt_path,
                tags=[dataset, aug, cname, f"severity{sev_idx}", "medmnist-c"],
            )

            acc, ece, mce, bal_acc, loss = evaluate(
                model,
                loader,
                device,
                do_ece=do_ece,
                do_bal_acc=do_bal_acc,
            )

            wandb.log({
                "test/acc": acc,
                "test/ece": ece,
                "test/mce": mce,
                "test/loss": loss,
                "test/bal_acc": bal_acc,
                "meta/stage": "stage3",
                "meta/dataset": dataset,
                "meta/model": model_name,
                "meta/augmentation": aug,
                "meta/corruption": cname,
                "meta/severity": sev_idx,
            })
            wandb.finish()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage-3 evaluation on clean + corruption benchmarks")
    parser.add_argument("dataset", choices=["cifar10", "cifar100", "dermamnist", "pathmnist"])
    parser.add_argument("aug", help="single augmentation token passed from runner")
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument(
        "--base",
        type=str,
        default=os.path.join(ROOT, "configs", "base", "stage3.yaml"),
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=os.path.join(ROOT, "configs", "models", "resnet18.yaml"),
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    aug = args.aug.lower()

    stage3_cfg = load_stage3_cfg(args.base)
    model_cfg = load_model_cfg(args.model_cfg)

    def run_one(a: str):
        print(f"[STAGE-3] dataset={dataset}, aug={a}")
        eval_clean(dataset, a, model_cfg, stage3_cfg)
        if dataset in ["cifar10", "cifar100"]:
            eval_cifar_c(dataset, a, model_cfg, stage3_cfg, args.severity)
        else:
            eval_medmnist_c(dataset, a, model_cfg, stage3_cfg, args.severity)

    run_one(aug)


if __name__ == "__main__":
    main()