# -*- coding: utf-8 -*-
"""
EN: Training entry for baseline & augmentation experiments (Stage 1 / 2).
    - Parses configs with command-line overrides
    - Sets seed / device
    - Builds loaders / model / optimizer
    - Optional EarlyStopping (only enabled in baseline.yaml for Stage-1)
    - Logs rich metrics to Weights & Biases (W&B)
    - Saves:
        * best.pt  (by val_acc, only when early_stopping.enabled = true)
        * last.pt  (always: checkpoint from the last finished epoch)

ZH: 训练主入口（用于 Stage-1 / Stage-2）：
    - 解析配置 + 命令行 override
    - 设定随机种子与设备
    - 构建数据加载器 / 模型 / 优化器
    - 可选早停（只在 baseline.yaml 中打开，用于 Stage-1）
    - 把丰富指标写入 W&B
    - 保存：
        * best.pt  （基于 val_acc，仅在 early_stopping.enabled = true 时）
        * last.pt  （始终保存：最后一个 epoch 的权重）
"""

import os
import torch
import wandb

from utils.config import parse_with_overrides
from utils.seed import seed_everything

# CIFAR loader
from xai_data.cifar import get_loaders as get_cifar_loaders
# MedMNIST loader
from xai_data.medmnist import get_loaders as get_medmnist_loaders

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
    """Safe numeric cast (handles YAML overrides)."""
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"{name} must be numeric, got {x} ({type(x)})") from e


def main():
    # -----------------------
    # 1) Load & seed config
    # -----------------------
    cfg = parse_with_overrides()
    seed_everything(cfg["seed"])

    # -----------------------
    # 2) Device
    # -----------------------
    wants_cuda = (cfg["device"] == "cuda")
    device = torch.device("cuda" if wants_cuda and torch.cuda.is_available() else "cpu")

    # -----------------------
    # 3) Build run name
    # -----------------------
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    data_name  = data_cfg.get("name", "unknown")
    aug_name   = data_cfg.get("aug", "noaug")
    model_name = model_cfg.get("name", "model")

    run_name = (
        f"{model_name}_{data_name}_"
        f"{aug_name}_{cfg['train']['exp_id']}_"
        f"lr{cfg['train']['lr']}_seed{cfg['seed']}"
    )

    wandb.init(
        project=cfg["wandb"]["project"],
        group=cfg["wandb"]["group"],
        name=run_name,
        config=cfg,
        settings=wandb.Settings(code_dir=".")
    )

    # -----------------------
    # 4) Data loaders
    # -----------------------
    # 注意：
    #   - Stage-2：CIFAR 使用 train / val split（val=eval，用于监控）
    #   - 真正的 test（clean + CIFAR-C）在 Stage-3 单独脚本里跑
    if cfg["data"]["name"].lower() in ["dermamnist", "pathmnist"]:
        train_ld, val_ld, test_ld, num_classes = get_medmnist_loaders(cfg)
    else:  # CIFAR
        train_ld, val_ld, num_classes = get_cifar_loaders(cfg)
        test_ld = None  # 占位，Stage-2 不用 test

    # -----------------------
    # 5) Model
    # -----------------------
    model = build_model(cfg["model"]["name"], num_classes).to(device)

    # -----------------------
    # 6) Optimizer
    # -----------------------
    lr           = _as_float(cfg["train"]["lr"], "train.lr")
    momentum     = _as_float(cfg["train"]["momentum"], "train.momentum")
    weight_decay = _as_float(cfg["train"]["weight_decay"], "train.weight_decay")

    optim = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -----------------------
    # 7) Early stopping (only Stage-1 uses it)
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
    # 8) Training Loop
    # -----------------------
    # best 只在 early_stopping 开启时才真正有意义（Stage-1）
    best_metric = -1e9 if cfg["early_stopping"].get("mode", "max") == "max" else 1e9

    os.makedirs(cfg["log"]["out_dir"], exist_ok=True)
    best_path = os.path.join(cfg["log"]["out_dir"], f"{run_name}_best.pt")
    last_path = os.path.join(cfg["log"]["out_dir"], f"{run_name}_last.pt")

    epochs = int(cfg["train"]["epochs"])

    for ep in range(epochs):
        # -----------------------
        # Train
        # -----------------------
        tr_acc, tr_loss = train_one_epoch(
            model,
            train_ld,
            optim,
            device,
            scaler,
            cfg_train=cfg["train"],
        )

        # -----------------------
        # Eval on validation loader (NOT test)
        # -----------------------
        use_bal_acc = cfg["eval"].get("do_bal_acc", False)

        va_acc, va_ece, va_mce, va_bal_acc, va_loss = evaluate(
            model,
            val_ld,
            device,
            do_ece=cfg["eval"]["do_ece"],
            do_bal_acc=use_bal_acc,
        )

        # -----------------------
        # Save LAST checkpoint (always)
        # -----------------------
        torch.save(model.state_dict(), last_path)

        # -----------------------
        # Save BEST checkpoint (only if ES enabled, e.g. Stage-1)
        # -----------------------
        if use_es:
            is_better = (
                va_acc > best_metric
                if cfg["early_stopping"].get("mode", "max") == "max"
                else va_acc < best_metric
            )
            if is_better:
                best_metric = va_acc
                torch.save(model.state_dict(), best_path)

        # -----------------------
        # Log to W&B
        # -----------------------
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

        # -----------------------
        # Early stop check (Stage-1 only)
        # -----------------------
        if use_es and es.step(va_acc):
            print(f"[EarlyStop] epoch={ep}, best_val_acc={best_metric:.4f}")
            break

    # -----------------------
    # Final messages + W&B artifacts
    # -----------------------
    print(f"[DONE] Last checkpoint saved to   {last_path}")
    wandb.save(last_path)

    if use_es and os.path.exists(best_path):
        print(f"[INFO] Best checkpoint (Stage-1) saved to {best_path}")
        wandb.save(best_path)

    wandb.finish()


if __name__ == "__main__":
    main()