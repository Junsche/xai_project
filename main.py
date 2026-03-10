# -*- coding: utf-8 -*-
"""
Training entry for baseline & augmentation experiments (Stage 1 / 2).

What this script does:
- Parses YAML configs with optional command-line overrides
- Sets random seed and selects device (CPU/CUDA)
- Builds data loaders, model, and optimizer
- Optional EarlyStopping (intended for Stage-1 baseline)
- Logs metrics to Weights & Biases (W&B)
- Saves checkpoints:
    * best.pt  (by val_acc, only when early_stopping.enabled = true)
    * last.pt  (always: checkpoint from the last finished epoch)
"""

import os
import torch
import wandb

from utils.config import parse_with_overrides
from utils.seed import seed_everything

# Dataset loaders
from xai_data.cifar import get_loaders as get_cifar_loaders
from xai_data.medmnist import get_loaders as get_medmnist_loaders

# Model + training utilities
from models.factory import build_model
from train.trainer import train_one_epoch, evaluate

from torch.optim import SGD


# ---------------------------
# EarlyStopping helper
# ---------------------------
class EarlyStopping:
    """Minimal early stopping helper."""

    def __init__(self, mode="max", patience=5, min_delta=0.0):
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad = 0
        self.should_stop = False

    def step(self, cur):
        """
        Update early-stopping state with the current metric value.

        Returns:
            bool: True if training should stop, otherwise False.
        """
        if self.best is None:
            self.best = cur
            return False

        improved = (
            cur > self.best + self.min_delta
            if self.mode == "max"
            else cur < self.best - self.min_delta
        )

        if improved:
            self.best = cur
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.should_stop = True

        return self.should_stop


def _as_float(x, name):
    """Safe numeric cast (useful when YAML overrides produce strings)."""
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"{name} must be numeric, got {x} ({type(x)})") from e


def main():
    # -----------------------
    # 1) Load config + seed
    # -----------------------
    cfg = parse_with_overrides()
    seed_everything(cfg["seed"])

    # -----------------------
    # 2) Device selection
    # -----------------------
    wants_cuda = (cfg["device"] == "cuda")
    device = torch.device("cuda" if wants_cuda and torch.cuda.is_available() else "cpu")

    # -----------------------
    # 3) Build a descriptive run name (for W&B and checkpoint files)
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

    # Initialize W&B logging for this run
    wandb.init(
        project=cfg["wandb"]["project"],
        group=cfg["wandb"]["group"],
        name=run_name,
        config=cfg,
        settings=wandb.Settings(code_dir=".")
    )

    # -----------------------
    # 4) Build data loaders
    # -----------------------
    # Note:
    # - For Stage-2, we typically monitor performance on a validation split.
    # - Final test evaluation (clean test and corruption test like CIFAR-C) is handled elsewhere (Stage-3).
    if cfg["data"]["name"].lower() in ["dermamnist", "pathmnist"]:
        train_ld, val_ld, test_ld, num_classes = get_medmnist_loaders(cfg)
    else:  # CIFAR family
        train_ld, val_ld, num_classes = get_cifar_loaders(cfg)
        test_ld = None  # Placeholder (this script does not use test for CIFAR in Stage-2)

    # -----------------------
    # 5) Build model
    # -----------------------
    model = build_model(cfg["model"], num_classes).to(device)

    # -----------------------
    # 6) Optimizer + AMP scaler
    # -----------------------
    lr = _as_float(cfg["train"]["lr"], "train.lr")
    momentum = _as_float(cfg["train"]["momentum"], "train.momentum")
    weight_decay = _as_float(cfg["train"]["weight_decay"], "train.weight_decay")

    optim = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -----------------------
    # 7) Early stopping (intended for Stage-1 baseline)
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
    # 8) Training loop
    # -----------------------
    # best checkpoint is meaningful only if early stopping is enabled
    best_metric = -1e9 if cfg["early_stopping"].get("mode", "max") == "max" else 1e9

    os.makedirs(cfg["log"]["out_dir"], exist_ok=True)
    best_path = os.path.join(cfg["log"]["out_dir"], f"{run_name}_best.pt")
    last_path = os.path.join(cfg["log"]["out_dir"], f"{run_name}_last.pt")

    epochs = int(cfg["train"]["epochs"])

    for ep in range(epochs):
        # 8.1) Train for one epoch
        tr_acc, tr_loss = train_one_epoch(
            model,
            train_ld,
            optim,
            device,
            scaler,
            cfg_train=cfg["train"],
        )

        # 8.2) Evaluate on validation loader (not test)
        use_bal_acc = cfg["eval"].get("do_bal_acc", False)

        va_acc, va_ece, va_mce, va_bal_acc, va_loss = evaluate(
            model,
            val_ld,
            device,
            do_ece=cfg["eval"]["do_ece"],
            do_bal_acc=use_bal_acc,
        )

        # 8.3) Save last checkpoint (always)
        torch.save(model.state_dict(), last_path)

        # 8.4) Save best checkpoint (only if early stopping enabled)
        if use_es:
            is_better = (
                va_acc > best_metric
                if cfg["early_stopping"].get("mode", "max") == "max"
                else va_acc < best_metric
            )
            if is_better:
                best_metric = va_acc
                torch.save(model.state_dict(), best_path)

        # 8.5) Log to W&B
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
        }

        if use_bal_acc and va_bal_acc is not None:
            log_dict["eval/bal_acc"] = va_bal_acc

        wandb.log(log_dict)

        # 8.6) Early stop check
        if use_es and es.step(va_acc):
            print(f"[EarlyStop] epoch={ep}, best_val_acc={best_metric:.4f}")
            break

    # -----------------------
    # 9) Final messages + W&B artifacts
    # -----------------------
    print(f"[DONE] Last checkpoint saved to   {last_path}")
    wandb.save(last_path)

    if use_es and os.path.exists(best_path):
        print(f"[INFO] Best checkpoint (Stage-1) saved to {best_path}")
        wandb.save(best_path)

    wandb.finish()


if __name__ == "__main__":
    main()