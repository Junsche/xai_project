# -*- coding: utf-8 -*-
"""
Training entry for Stage-1 / Stage-2 experiments.

What this script does:
- Parses layered YAML configs with optional command-line overrides
- Sets random seed and selects device
- Builds loaders, model, optimizer
- Supports optional EarlyStopping (currently mainly for Stage-1 if enabled)
- Logs metrics to Weights & Biases (W&B)
- Saves checkpoints:
    * best.pt  (only when early_stopping.enabled = true)
    * last.pt  (always)
"""

import os
import sys
import torch
import wandb
from torch.optim import SGD

from utils.config import parse_with_overrides
from utils.seed import seed_everything

# Data loaders
from data_modules.cifar import get_loaders as get_cifar_loaders
from data_modules.medmnist import get_loaders as get_medmnist_loaders

# Model + training
from models.factory import build_model
from train.trainer import train_one_epoch, evaluate
from train.early_stopping import EarlyStopping


def _as_float(x, name):
    """Safe numeric cast."""
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"{name} must be numeric, got {x} ({type(x)})") from e


def _infer_stage_from_configs(config_paths):
    """
    Infer stage from config file path names.
    """
    joined = " ".join(str(p).lower() for p in config_paths)

    if "configs/base/stage1.yaml" in joined:
        return "stage1"
    if "configs/base/stage2.yaml" in joined:
        return "stage2"
    if "configs/base/stage3.yaml" in joined:
        return "stage3"
    return "unknown"


def main():
    # -----------------------
    # 1) Load config + seed
    # -----------------------
    cfg = parse_with_overrides()
    seed_everything(cfg["seed"])

    raw_config_paths = [arg for arg in sys.argv[1:] if arg.endswith(".yaml")]
    inferred_stage = _infer_stage_from_configs(raw_config_paths)
    cfg["stage"] = cfg.get("stage", inferred_stage)

    # -----------------------
    # 2) Device selection
    # -----------------------
    wants_cuda = (cfg["device"] == "cuda")
    device = torch.device("cuda" if wants_cuda and torch.cuda.is_available() else "cpu")

    # -----------------------
    # 3) Validate optimizer choice
    # -----------------------
    optimizer_name = str(cfg["train"].get("optimizer", "sgd")).lower()
    if optimizer_name != "sgd":
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. "
            f"Current implementation only supports SGD."
        )

    # -----------------------
    # 4) Build descriptive run name
    # -----------------------
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    data_name = data_cfg.get("name", "unknown")
    aug_name = data_cfg.get("aug", "noaug")
    model_name = model_cfg.get("name", "model")

    run_name = (
        f"{model_name}_{data_name}_"
        f"{aug_name}_{cfg['train']['exp_id']}_"
        f"lr{cfg['train']['lr']}_seed{cfg['seed']}"
    )

    # -----------------------
    # 5) Initialize W&B
    # -----------------------
    wandb_cfg = dict(cfg)
    wandb_cfg["stage"] = cfg["stage"]

    wandb.init(
        project=cfg["wandb"]["project"],
        group=cfg["wandb"]["group"],
        name=run_name,
        tags=cfg["wandb"].get("tags", []),
        config=wandb_cfg,
        settings=wandb.Settings(code_dir="."),
    )

    # -----------------------
    # 6) Build data loaders
    # -----------------------
    dataset_name = str(cfg["data"]["name"]).lower()

    if dataset_name in ["dermamnist", "pathmnist"]:
        train_ld, val_ld, test_ld, num_classes = get_medmnist_loaders(cfg)
    else:
        train_ld, val_ld, num_classes = get_cifar_loaders(cfg)
        test_ld = None  # not used here

    # -----------------------
    # 7) Build model
    # -----------------------
    model = build_model(cfg["model"], num_classes).to(device)

    # -----------------------
    # 8) Optimizer + AMP scaler
    # -----------------------
    lr = _as_float(cfg["train"]["lr"], "train.lr")
    momentum = _as_float(cfg["train"]["momentum"], "train.momentum")
    weight_decay = _as_float(cfg["train"]["weight_decay"], "train.weight_decay")

    optim = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -----------------------
    # 9) Early stopping
    # -----------------------
    use_es = bool(cfg["early_stopping"].get("enabled", False))
    es = None
    if use_es:
        es = EarlyStopping(
            mode=cfg["early_stopping"].get("mode", "max"),
            patience=cfg["early_stopping"].get("patience", 5),
            min_delta=cfg["early_stopping"].get("min_delta", 0.0),
        )

    # -----------------------
    # 10) Checkpoint paths
    # -----------------------
    out_dir = cfg["log"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    best_path = os.path.join(out_dir, f"{run_name}_best.pt")
    last_path = os.path.join(out_dir, f"{run_name}_last.pt")

    monitor_mode = cfg["early_stopping"].get("mode", "max")
    best_metric = -1e9 if monitor_mode == "max" else 1e9

    # -----------------------
    # 11) Training loop
    # -----------------------
    epochs = int(cfg["train"]["epochs"])
    use_bal_acc = bool(cfg["eval"].get("do_bal_acc", False))

    for ep in range(epochs):
        # Train
        tr_acc, tr_loss = train_one_epoch(
            model,
            train_ld,
            optim,
            device,
            scaler,
            cfg_train=cfg["train"],
        )

        # Eval on validation-like loader
        va_acc, va_ece, va_mce, va_bal_acc, va_loss = evaluate(
            model,
            val_ld,
            device,
            do_ece=cfg["eval"].get("do_ece", True),
            do_bal_acc=use_bal_acc,
        )

        # Always save last
        torch.save(model.state_dict(), last_path)

        # Save best only if early stopping enabled
        if use_es:
            is_better = (
                va_acc > best_metric if monitor_mode == "max" else va_acc < best_metric
            )
            if is_better:
                best_metric = va_acc
                torch.save(model.state_dict(), best_path)

        # Log
        log_dict = {
            "epoch": ep,
            "train/acc": tr_acc,
            "train/loss": tr_loss,
            "eval/acc": va_acc,
            "eval/loss": va_loss,
            "eval/ece": va_ece,
            "eval/mce": va_mce,
            "lr": optim.param_groups[0]["lr"],
            "early_stop_enabled": int(use_es),
            "meta/stage": cfg["stage"],
            "meta/dataset": data_name,
            "meta/model": model_name,
            "meta/augmentation": aug_name,
            "meta/exp_id": cfg["train"]["exp_id"],
        }

        if use_bal_acc and va_bal_acc is not None:
            log_dict["eval/bal_acc"] = va_bal_acc

        wandb.log(log_dict)

        # Early stop check
        if use_es and es.step(va_acc):
            print(f"[EarlyStop] epoch={ep}, best_val_acc={best_metric:.4f}")
            break

    # -----------------------
    # 12) Final messages
    # -----------------------
    print(f"[DONE] Last checkpoint saved to   {last_path}")
    wandb.save(last_path)

    if use_es and os.path.exists(best_path):
        print(f"[INFO] Best checkpoint saved to {best_path}")
        wandb.save(best_path)

    wandb.finish()


if __name__ == "__main__":
    main()