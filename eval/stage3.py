# -*- coding: utf-8 -*-
"""
Stage-3: Clean Test + Corruption Robustness Evaluation
- CIFAR10/100: clean test + CIFAR-C
- MedMNIST (DermaMNIST / PathMNIST): clean test + MedMNIST-C

W&B logs:
  test/acc, test/ece, test/mce, test/loss, test/bal_acc
  meta/corruption, meta/severity

IMPORTANT:
- No pretrained downloads in Stage-3 (weights='none'); we load Stage-2 checkpoints.
- Supports model-aware preprocessing via model.input_size / model.preprocessing.
- Stage-2 checkpoint contract (lr) is loaded from configs/protocol/stage2_selected_lrs.yaml.

DEBUG VERSION:
- Adds detailed debug prints
- Frees cache after clean eval
- Uses streaming CIFAR-C loading: one corruption + one severity at a time
"""

import os
import sys
import gc
import argparse
from typing import Optional, Any, List, Tuple

import yaml
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# ---------------------------------------------------------------------
# Ensure imports from project root
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factory import build_model
from train.trainer import evaluate
from data_modules.transforms_registry import REGISTRY
from utils.seed import seed_everything

from data_modules.medmnist import get_loaders as get_medmnist_loaders
from data_modules.medmnist_c import make_medmnist_c_loader


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

WANDB_PROJECT = "stage3-robustness-v4"
SEED = 1437
CKPT_DIR = "./runs"
STAGE2_LR_CFG = os.path.join(ROOT, "configs", "protocol", "stage2_selected_lrs.yaml")

DEBUG_STAGE3 = True


def dbg(msg: str):
    if DEBUG_STAGE3:
        print(f"[DEBUG][STAGE3] {msg}", flush=True)


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


def load_model_cfg(path: str) -> dict:
    cfg = load_yaml(path)
    model_cfg = dict(cfg.get("model", {}))
    if "name" not in model_cfg:
        raise KeyError(f"`model.name` missing in {path}")
    model_cfg.setdefault("weights", "none")
    return model_cfg


def load_stage2_lrs() -> dict:
    return load_yaml(STAGE2_LR_CFG)


def get_stage2_selected_lr(model_name: str, dataset: str) -> str:
    lr_all = load_stage2_lrs()

    if model_name not in lr_all:
        raise KeyError(f"No Stage-2 LR block found for model: {model_name}")
    if dataset not in lr_all[model_name]:
        raise KeyError(f"No Stage-2 LR found for model={model_name}, dataset={dataset}")

    lr = lr_all[model_name][dataset]
    if lr is None:
        raise ValueError(f"Stage-2 LR is None for model={model_name}, dataset={dataset}")

    return str(lr)


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
# Checkpoint path (must match Stage-2 naming)
# ============================================================

def build_ckpt_path(dataset: str, aug: str, model_name: str) -> str:
    lr = get_stage2_selected_lr(model_name, dataset)
    aug_token = "baseline" if aug in ["mixup", "cutmix"] else aug
    exp_id = f"S2_{aug}"
    run_name = f"{model_name}_{dataset}_{aug_token}_{exp_id}_lr{lr}_seed{SEED}"
    return os.path.join(CKPT_DIR, f"{run_name}_last.pt")


# ============================================================
# Model loading
# ============================================================

def build_model_for_stage3(model_cfg: dict, num_classes: int, device: torch.device):
    cfg = dict(model_cfg)
    cfg["weights"] = "none"
    dbg(f"building model: name={cfg.get('name')} num_classes={num_classes} device={device}")
    model = build_model(cfg, num_classes).to(device)
    dbg("model built successfully")
    return model


def load_ckpt(model, ckpt_path: str, device: torch.device):
    dbg(f"loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    dbg("checkpoint loaded into memory")
    model.load_state_dict(state)
    dbg("state_dict loaded into model")
    model.eval()
    dbg("model set to eval()")


def cleanup_after_eval(device: torch.device):
    dbg("cleanup_after_eval(): start")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        dbg("torch.cuda.empty_cache() called")
    dbg("cleanup_after_eval(): done")


# ============================================================
# W&B helpers
# ============================================================

def wandb_run(
    *,
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

    dbg(f"wandb.init(): corruption={corruption}, severity={severity}")
    wandb.init(
        project=WANDB_PROJECT,
        group=f"{dataset}_{aug}",
        name=f"{model_name}_{dataset}_{aug}_{corruption}_s{severity}",
        tags=tags or [dataset, aug, corruption, f"severity{severity}"],
        config=cfg,
        settings=wandb.Settings(code_dir="."),
    )


# ============================================================
# Data loaders: CIFAR clean
# ============================================================

def make_clean_test_loader_cifar(dataset: str, data_cfg: dict, model_cfg: dict) -> DataLoader:
    dbg(f"make_clean_test_loader_cifar(): dataset={dataset}")

    if dataset == "cifar10":
        ds_cls = datasets.CIFAR10
    elif dataset == "cifar100":
        ds_cls = datasets.CIFAR100
    else:
        raise ValueError(dataset)

    mean = _as_tuple_floats(data_cfg.get("mean"), (0.5, 0.5, 0.5))
    std = _as_tuple_floats(data_cfg.get("std"), (0.5, 0.5, 0.5))

    model_input_size = int(model_cfg.get("input_size", data_cfg.get("img_size", 32)))
    preprocessing = str(model_cfg.get("preprocessing", "cifar")).lower()

    dbg(f"clean CIFAR transform: input_size={model_input_size}, preprocessing={preprocessing}")

    _, test_tf = REGISTRY["baseline"](
        img_size=model_input_size,
        mean=mean,
        std=std,
        preprocessing=preprocessing,
    )

    ds = ds_cls(
        root=data_cfg["root"],
        train=False,
        download=False,
        transform=test_tf,
    )

    dbg(f"clean CIFAR dataset size={len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    dbg("clean CIFAR DataLoader built")
    return loader


# ============================================================
# Data loaders: CIFAR-C
# ============================================================

def _normalize_batch_nchw(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def _apply_eval_resize_nchw(
    x: torch.Tensor,
    img_size: int,
    preprocessing: str = "cifar",
) -> torch.Tensor:
    preprocessing = str(preprocessing).lower()

    if preprocessing == "imagenet":
        resize_size = max(img_size, int(round(img_size * 256 / 224)))
        x = F.interpolate(x, size=(resize_size, resize_size), mode="bilinear", align_corners=False)

        top = (resize_size - img_size) // 2
        left = (resize_size - img_size) // 2
        x = x[:, :, top:top + img_size, left:left + img_size]
        return x

    if x.shape[-2] != img_size or x.shape[-1] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    return x


def make_cifar_c_loader(
    root: str,
    *,
    corruption: str,
    severity: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    img_size: int,
    preprocessing: str = "cifar",
    batch_size=128,
    num_workers=0,
):
    """
    Build only ONE CIFAR-C loader for one corruption + one severity.
    This avoids holding all corruptions and all severities in memory at once.
    """
    root = os.path.abspath(root)

    if not (1 <= severity <= 5):
        raise ValueError(f"severity must be 1..5, got {severity}")

    label_path = os.path.join(root, "labels.npy")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"CIFAR-C labels.npy not found: {label_path}")

    corr_path = os.path.join(root, f"{corruption}.npy")
    if not os.path.isfile(corr_path):
        raise FileNotFoundError(f"CIFAR-C corruption file not found: {corr_path}")

    dbg(f"make_cifar_c_loader(): root={root}")
    dbg(f"make_cifar_c_loader(): corruption={corruption}, severity={severity}")

    y_all = np.load(label_path)
    y_all = np.asarray(y_all)
    y_len = int(y_all.shape[0])
    dbg(f"labels.npy shape={y_all.shape}, len={y_len}")

    x_all = np.load(corr_path)
    dbg(f"{corruption}: raw array shape={x_all.shape}, dtype={x_all.dtype}")

    s = severity - 1
    xs = x_all[s * 10000:(s + 1) * 10000]

    if y_len == 10000:
        ys = y_all
    else:
        ys = y_all[s * 10000:(s + 1) * 10000]

    if xs.ndim == 4:
        xs = xs.transpose(0, 3, 1, 2)

    dbg(f"{corruption} severity {severity}: xs after transpose shape={xs.shape}")

    xs = torch.from_numpy(xs).float() / 255.0
    xs = _apply_eval_resize_nchw(
        xs,
        img_size=img_size,
        preprocessing=preprocessing,
    )
    xs = _normalize_batch_nchw(xs, mean, std)

    ys = torch.from_numpy(np.asarray(ys)).long()

    dbg(f"{corruption} severity {severity}: tensor xs shape={tuple(xs.shape)}, ys shape={tuple(ys.shape)}")

    ds = TensorDataset(xs, ys)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    dbg(f"{corruption} severity {severity}: dataloader built")

    del x_all
    gc.collect()

    return dl


# ============================================================
# Data loaders: MedMNIST clean
# ============================================================

def make_clean_test_loader_medmnist(dataset_cfg: dict, model_cfg: dict) -> DataLoader:
    dbg("make_clean_test_loader_medmnist()")
    cfg = {
        "data": dict(dataset_cfg["data"]),
        "model": dict(model_cfg),
    }
    cfg["data"].setdefault("aug", "baseline")
    _, _, test_ld, _ = get_medmnist_loaders(cfg)
    dbg("clean MedMNIST DataLoader built")
    return test_ld


# ============================================================
# Eval: Clean
# ============================================================

def eval_clean(dataset: str, aug: str, model_cfg: dict):
    dbg(f"eval_clean(): dataset={dataset}, aug={aug}")

    dataset_cfg = load_dataset_cfg(dataset)
    data_cfg = dataset_cfg["data"]
    eval_cfg = dataset_cfg.get("eval", {})

    do_ece = bool(eval_cfg.get("do_ece", True))
    do_bal_acc = bool(eval_cfg.get("do_bal_acc", False))

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbg(f"eval_clean(): device={device}")

    model_name = model_cfg["name"]
    ckpt_path = build_ckpt_path(dataset, aug, model_name)
    dbg(f"eval_clean(): ckpt_path={ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[CLEAN] Checkpoint not found: {ckpt_path}")

    num_classes = int(dataset_cfg["model"]["num_classes"])

    if dataset in ["cifar10", "cifar100"]:
        dbg("eval_clean(): building CIFAR clean loader")
        loader = make_clean_test_loader_cifar(dataset, data_cfg, model_cfg)
    elif dataset in ["dermamnist", "pathmnist"]:
        dbg("eval_clean(): building MedMNIST clean loader")
        loader = make_clean_test_loader_medmnist(dataset_cfg, model_cfg)
    else:
        raise ValueError(dataset)

    dbg("eval_clean(): building model")
    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    dbg("eval_clean(): starting wandb clean run")
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
            "model": {"name": model_name, "weights": "none"},
        },
        settings=wandb.Settings(code_dir="."),
    )

    dbg("eval_clean(): entering evaluate()")
    acc, ece, mce, bal_acc, loss = evaluate(
        model,
        loader,
        device,
        do_ece=do_ece,
        do_bal_acc=do_bal_acc,
    )
    dbg("eval_clean(): evaluate() finished")

    wandb.log({
        "test/acc": acc,
        "test/ece": ece,
        "test/mce": mce,
        "test/loss": loss,
        "test/bal_acc": bal_acc,
        "meta/corruption": "clean",
        "meta/severity": 0,
    })
    dbg("eval_clean(): wandb.log() finished")
    wandb.finish()
    dbg("eval_clean(): wandb.finish() finished")

    del loader
    del model
    cleanup_after_eval(device)


# ============================================================
# Eval: CIFAR-C
# ============================================================

def eval_cifar_c(dataset: str, aug: str, model_cfg: dict, severity: Optional[int]):
    dbg(f"eval_cifar_c(): dataset={dataset}, aug={aug}, severity={severity}")

    dataset_cfg = load_dataset_cfg(dataset)
    data_cfg = dataset_cfg["data"]
    eval_cfg = dataset_cfg.get("eval", {})

    do_ece = bool(eval_cfg.get("do_ece", True))
    do_bal_acc = bool(eval_cfg.get("do_bal_acc", False))

    if severity is not None and not (1 <= severity <= 5):
        raise ValueError("--severity must be 1..5")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbg(f"eval_cifar_c(): device={device}")

    model_name = model_cfg["name"]
    ckpt_path = build_ckpt_path(dataset, aug, model_name)
    dbg(f"eval_cifar_c(): ckpt_path={ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[CIFAR-C] Checkpoint not found: {ckpt_path}")

    num_classes = int(dataset_cfg["model"]["num_classes"])
    dbg("eval_cifar_c(): building model")
    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    mean = _as_tuple_floats(data_cfg.get("mean"), (0.5, 0.5, 0.5))
    std = _as_tuple_floats(data_cfg.get("std"), (0.5, 0.5, 0.5))
    model_input_size = int(model_cfg.get("input_size", data_cfg.get("img_size", 32)))
    preprocessing = str(model_cfg.get("preprocessing", "cifar")).lower()

    wanted_corruptions = _get_list(data_cfg, "corruptions")
    wanted_severities = _get_list(data_cfg, "severities")

    cifar_c_root = data_cfg["cifar_c_root"]

    if wanted_corruptions is None:
        wanted_corruptions = sorted([
            fname.replace(".npy", "")
            for fname in os.listdir(cifar_c_root)
            if fname.endswith(".npy") and fname != "labels.npy"
        ])

    if wanted_severities is None:
        wanted_severities = [1, 2, 3, 4, 5]

    dbg("eval_cifar_c(): entering corruption loop")
    for cname in wanted_corruptions:
        dbg(f"eval_cifar_c(): corruption={cname}")

        for sev_idx in wanted_severities:
            dbg(f"eval_cifar_c(): corruption={cname}, severity={sev_idx}")

            if severity is not None and sev_idx != severity:
                dbg(f"eval_cifar_c(): skipping severity={sev_idx} (filtered by CLI severity)")
                continue

            loader = make_cifar_c_loader(
                cifar_c_root,
                corruption=cname,
                severity=sev_idx,
                mean=mean,
                std=std,
                img_size=model_input_size,
                preprocessing=preprocessing,
                batch_size=int(data_cfg.get("batch_size", 128)),
                num_workers=0,
            )

            wandb_run(
                dataset=dataset,
                aug=aug,
                model_name=model_name,
                corruption=cname,
                severity=sev_idx,
                ckpt_path=ckpt_path,
                tags=[dataset, aug, cname, f"severity{sev_idx}", "cifar-c"],
            )

            dbg(f"eval_cifar_c(): calling evaluate() for corruption={cname}, severity={sev_idx}")
            acc, ece, mce, bal_acc, loss = evaluate(
                model,
                loader,
                device,
                do_ece=do_ece,
                do_bal_acc=do_bal_acc,
            )
            dbg(f"eval_cifar_c(): evaluate() finished for corruption={cname}, severity={sev_idx}")

            wandb.log({
                "test/acc": acc,
                "test/ece": ece,
                "test/mce": mce,
                "test/loss": loss,
                "test/bal_acc": bal_acc,
                "meta/corruption": cname,
                "meta/severity": sev_idx,
            })
            dbg(f"eval_cifar_c(): wandb.log() done for corruption={cname}, severity={sev_idx}")
            wandb.finish()
            dbg(f"eval_cifar_c(): wandb.finish() done for corruption={cname}, severity={sev_idx}")

            del loader
            cleanup_after_eval(device)

    del model
    cleanup_after_eval(device)


# ============================================================
# Eval: MedMNIST-C
# ============================================================

def eval_medmnist_c(dataset: str, aug: str, model_cfg: dict, severity: Optional[int]):
    dbg(f"eval_medmnist_c(): dataset={dataset}, aug={aug}, severity={severity}")

    dataset_cfg = load_dataset_cfg(dataset)
    data_cfg = dataset_cfg["data"]
    eval_cfg = dataset_cfg.get("eval", {})

    do_ece = bool(eval_cfg.get("do_ece", True))
    do_bal_acc = bool(eval_cfg.get("do_bal_acc", True))

    if severity is not None and not (1 <= severity <= 5):
        raise ValueError("--severity must be 1..5")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbg(f"eval_medmnist_c(): device={device}")

    model_name = model_cfg["name"]
    ckpt_path = build_ckpt_path(dataset, aug, model_name)
    dbg(f"eval_medmnist_c(): ckpt_path={ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[MedMNIST-C] Checkpoint not found: {ckpt_path}")

    num_classes = int(dataset_cfg["model"]["num_classes"])
    dbg("eval_medmnist_c(): building model")
    model = build_model_for_stage3(model_cfg, num_classes, device)
    load_ckpt(model, ckpt_path, device)

    c_root = data_cfg["medmnist_c_root"]
    img_size = int(model_cfg.get("input_size", data_cfg.get("img_size", 28)))
    preprocessing = str(model_cfg.get("preprocessing", "cifar")).lower()
    mean = _as_tuple_floats(data_cfg.get("mean"), (0.5, 0.5, 0.5))
    std = _as_tuple_floats(data_cfg.get("std"), (0.5, 0.5, 0.5))

    corruptions = data_cfg.get("corruptions", None)
    if corruptions is None:
        raise KeyError(f"{dataset}.yaml missing data.corruptions for MedMNIST-C.")

    severities = data_cfg.get("severities", [1, 2, 3, 4, 5])

    dbg("eval_medmnist_c(): entering corruption loop")
    for cname in corruptions:
        dbg(f"eval_medmnist_c(): corruption={cname}")
        for sev_idx in severities:
            dbg(f"eval_medmnist_c(): corruption={cname}, severity={sev_idx}")

            if severity is not None and sev_idx != severity:
                dbg(f"eval_medmnist_c(): skipping severity={sev_idx} (filtered by CLI severity)")
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
                preprocessing=preprocessing,
            )
            dbg(f"eval_medmnist_c(): loader built for corruption={cname}, severity={sev_idx}")

            wandb_run(
                dataset=dataset,
                aug=aug,
                model_name=model_name,
                corruption=cname,
                severity=sev_idx,
                ckpt_path=ckpt_path,
                tags=[dataset, aug, cname, f"severity{sev_idx}", "medmnist-c"],
            )

            dbg(f"eval_medmnist_c(): calling evaluate() for corruption={cname}, severity={sev_idx}")
            acc, ece, mce, bal_acc, loss = evaluate(
                model,
                loader,
                device,
                do_ece=do_ece,
                do_bal_acc=do_bal_acc,
            )
            dbg(f"eval_medmnist_c(): evaluate() finished for corruption={cname}, severity={sev_idx}")

            wandb.log({
                "test/acc": acc,
                "test/ece": ece,
                "test/mce": mce,
                "test/loss": loss,
                "test/bal_acc": bal_acc,
                "meta/corruption": cname,
                "meta/severity": sev_idx,
            })
            dbg(f"eval_medmnist_c(): wandb.log() done for corruption={cname}, severity={sev_idx}")
            wandb.finish()
            dbg(f"eval_medmnist_c(): wandb.finish() done for corruption={cname}, severity={sev_idx}")

            del loader
            cleanup_after_eval(device)

    del model
    cleanup_after_eval(device)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage-3 evaluation on clean + corruption benchmarks")
    parser.add_argument("dataset", choices=["cifar10", "cifar100", "dermamnist", "pathmnist"])
    parser.add_argument("aug", nargs="?", default=None)
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=os.path.join(ROOT, "configs", "models", "resnet18.yaml"),
        help="Path to model YAML, e.g. configs/models/resnet18.yaml or configs/models/vit_b.yaml",
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    aug = args.aug.lower() if args.aug is not None else None

    model_cfg = load_model_cfg(args.model_cfg)

    def run_one(a: str):
        print(f"[STAGE-3] dataset={dataset}, model={model_cfg['name']}, aug={a}", flush=True)
        dbg("run_one(): starting clean eval")
        eval_clean(dataset, a, model_cfg)
        dbg("run_one(): clean eval finished")

        if dataset in ["cifar10", "cifar100"]:
            dbg("run_one(): entering CIFAR-C eval")
            eval_cifar_c(dataset, a, model_cfg, args.severity)
            dbg("run_one(): CIFAR-C eval finished")
        else:
            dbg("run_one(): entering MedMNIST-C eval")
            eval_medmnist_c(dataset, a, model_cfg, args.severity)
            dbg("run_one(): MedMNIST-C eval finished")

    if aug is None:
        for a in DEFAULT_AUGS:
            run_one(a)
    else:
        run_one(aug)


if __name__ == "__main__":
    main()