# -*- coding: utf-8 -*-
"""
EN: Training entry for baseline & aug experiments (Stage 1/2/3).
    - Parses configs with command-line overrides
    - Sets seed/cuda
    - Builds loaders/model/optimizer
    - Supports EarlyStopping (enabled only for baseline YAML)
    - Logs to Weights & Biases (W&B)
    - Saves best checkpoint by validation metric

ZH: 训练主入口（适用于基线与增强对比的三阶段）：
    - 解析配置 + 命令行 override
    - 设定随机种子与设备
    - 构建数据加载器 / 模型 / 优化器
    - 支持早停（只在 baseline.yaml 中启用）
    - 记录到 W&B
    - 根据验证集指标保存最优权重
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
        improved = (cur > self.best + self.min_delta) if self.mode == "max" else (cur < self.best - self.min_delta)
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
    run_name = (
        f"{cfg['model']['name']}_{cfg['data']['name']}_"
        f"{cfg['data']['aug']}_{cfg['train']['exp_id']}_"
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
    if cfg["data"]["name"].lower() in ["dermamnist", "pathmnist"]:
        train_ld, val_ld, test_ld, num_classes = get_medmnist_loaders(cfg)
    else:  # CIFAR
        train_ld, val_ld, num_classes = get_cifar_loaders(cfg)
        test_ld = None

    # -----------------------
    # 5) Model
    # -----------------------
    model = build_model(cfg["model"]["name"], num_classes).to(device)

    # -----------------------
    # 6) Optimizer
    # -----------------------
    lr = _as_float(cfg["train"]["lr"], "train.lr")
    momentum = _as_float(cfg["train"]["momentum"], "train.momentum")
    weight_decay = _as_float(cfg["train"]["weight_decay"], "train.weight_decay")

    optim = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # -----------------------
    # 7) Early stopping
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
    best_metric = -1e9 if cfg["early_stopping"]["mode"] == "max" else 1e9

    os.makedirs(cfg["log"]["out_dir"], exist_ok=True)
    best_path = os.path.join(cfg["log"]["out_dir"], f"{run_name}_best.pt")

    epochs = int(cfg["train"]["epochs"])

    for ep in range(epochs):

        # Train
        tr_acc = train_one_epoch(model, train_ld, optim, device, scaler, cfg_train=cfg["train"])

        # Eval
        use_bal_acc = cfg["eval"].get("do_bal_acc", False)  # CIFAR=False, MedMNIST=True

        va_acc, va_ece, va_bal_acc = evaluate(
            model,
            val_ld,
            device,
            do_ece=cfg["eval"]["do_ece"],
            do_bal_acc=use_bal_acc,
        )

        # -----------------------
        #  Log to W&B
        # -----------------------
        log_dict = {
            "epoch": ep,
            "train/acc": tr_acc,
            "eval/acc": va_acc,
            "eval/ece": va_ece if va_ece is not None else None,
            "lr": optim.param_groups[0]["lr"],
            "early_stop_enabled": int(use_es),
        }

        if use_bal_acc:
            log_dict["eval/bal_acc"] = va_bal_acc

        wandb.log(log_dict)

        # -----------------------
        # Save best model (by val_acc)
        # -----------------------
        is_better = (
            va_acc > best_metric
            if cfg["early_stopping"]["mode"] == "max"
            else va_acc < best_metric
        )

        if is_better:
            best_metric = va_acc
            torch.save(model.state_dict(), best_path)

        # -----------------------
        # Early stop check
        # -----------------------
        if use_es and es.step(va_acc):
            print(f"[EarlyStop] epoch={ep}, best_val_acc={best_metric:.4f}")
            break

    print(f"[DONE] Best checkpoint saved to {best_path}")
    wandb.save(best_path)
    wandb.finish()


if __name__ == "__main__":
    main()