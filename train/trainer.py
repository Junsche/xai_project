# train/trainer.py
# -*- coding: utf-8 -*-
"""
EN:
    Training + evaluation utilities used by Stage-1/Stage-2 (main.py),
    and also reused by Stage-3 evaluation scripts.

    Exports (IMPORTANT for main.py):
        - train_one_epoch(model, loader, optimizer, device, scaler=None, cfg_train=None)
        - evaluate(model, loader, device, do_ece=True, do_bal_acc=False)

    Supports:
        - MixUp / CutMix (batch-level) controlled via cfg_train keys:
            * cfg_train["mixup_alpha"]
            * cfg_train["cutmix_alpha"]
        - ECE / MCE computation (confidence calibration)
        - Balanced Accuracy (optional, for imbalanced datasets)
"""

import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .metrics import accuracy


# ------------------------------------------------------------
# Batch-level augmentations: MixUp / CutMix
# ------------------------------------------------------------
def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    y_mix = (y, y[idx], lam)
    return x_mix, y_mix


def _cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = x.size()

    cx, cy = random.randrange(W), random.randrange(H)
    bw = int(W * (1.0 - lam) ** 0.5)
    bh = int(H * (1.0 - lam) ** 0.5)

    x1, y1 = max(cx - bw // 2, 0), max(cy - bh // 2, 0)
    x2, y2 = min(cx + bw // 2, W), min(cy + bh // 2, H)

    idx = torch.randperm(B, device=x.device)
    x2v = x.clone()
    x2v[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    # Effective lambda based on the actually replaced area
    lam_eff = 1.0 - (x2 - x1) * (y2 - y1) / float(W * H)
    y_mix = (y, y[idx], lam_eff)
    return x2v, y_mix


def _criterion(logits: torch.Tensor, target):
    """
    EN:
        If target is a tuple (y1, y2, lam), compute mix loss.
        Else standard cross-entropy.
    """
    if isinstance(target, tuple):
        y1, y2, lam = target
        return lam * F.cross_entropy(logits, y1) + (1.0 - lam) * F.cross_entropy(logits, y2)
    return F.cross_entropy(logits, target)


# ------------------------------------------------------------
# Evaluation: Acc + Loss + (optional) ECE/MCE + (optional) BalAcc
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    do_ece: bool = True,
    do_bal_acc: bool = False,
    n_bins: int = 15,
):
    """
    Returns:
        acc, ece, mce, bal_acc, avg_loss

    Notes:
        - ECE/MCE are computed from logits and labels across the full loader.
        - Balanced Accuracy uses sklearn if enabled.
    """
    model.eval()

    tot_correct = 0
    tot_loss = 0.0
    tot_n = 0

    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        bs = x.size(0)

        tot_correct += correct
        tot_loss += loss.item() * bs
        tot_n += bs

        if do_ece or do_bal_acc:
            all_logits.append(logits.detach())
            all_labels.append(y.detach())

    acc = tot_correct / max(1, tot_n)
    avg_loss = tot_loss / max(1, tot_n)

    ece = None
    mce = None
    bal_acc = None

    if (do_ece or do_bal_acc) and len(all_logits) > 0:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        # --- ECE + MCE (manual implementation) ---
        if do_ece:
            probs = F.softmax(logits_cat, dim=1)
            confs, preds = probs.max(dim=1)
            correct_vec = (preds == labels_cat).float()

            ece_tensor = torch.zeros(1, device=device)
            mce_tensor = torch.zeros(1, device=device)

            bins = torch.linspace(0, 1, steps=n_bins + 1, device=device)

            for i in range(n_bins):
                lo, hi = bins[i], bins[i + 1]
                mask = (confs > lo) & (confs <= hi)
                m = mask.sum()

                if m == 0:
                    continue

                conf_bin = confs[mask].mean()
                acc_bin = correct_vec[mask].mean()
                gap = torch.abs(conf_bin - acc_bin)

                ece_tensor += (m / len(confs)) * gap
                mce_tensor = torch.max(mce_tensor, gap)

            ece = ece_tensor.item()
            mce = mce_tensor.item()

        # --- Balanced Accuracy (sklearn) ---
        if do_bal_acc:
            from sklearn.metrics import balanced_accuracy_score
            _, preds = logits_cat.max(dim=1)
            bal_acc = balanced_accuracy_score(
                labels_cat.cpu().numpy(),
                preds.cpu().numpy(),
            )

    return acc, ece, mce, bal_acc, avg_loss


# ------------------------------------------------------------
# Training: one epoch
# ------------------------------------------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    cfg_train: Optional[dict] = None,
):
    """
    EN:
        Run one training epoch.
        - Applies MixUp/CutMix if cfg_train includes mixup_alpha / cutmix_alpha.
        - Supports AMP if scaler is provided (cuda).

    Returns:
        avg_acc, avg_loss
    """
    model.train()

    tot_acc = 0.0
    tot_loss = 0.0
    tot_n = 0

    use_mixup = cfg_train is not None and "mixup_alpha" in cfg_train
    use_cutmix = cfg_train is not None and "cutmix_alpha" in cfg_train

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Batch-level mixing (optional)
        if use_mixup:
            x, y = _mixup(x, y, float(cfg_train["mixup_alpha"]))
        if use_cutmix:
            x, y = _cutmix(x, y, float(cfg_train["cutmix_alpha"]))

        # Forward/backward
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = _criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = _criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)

        # For MixUp/CutMix accuracy, use the "original" labels y1
        acc_target = y if not isinstance(y, tuple) else y[0]
        tot_acc += accuracy(logits, acc_target) * bs
        tot_loss += loss.item() * bs
        tot_n += bs

    avg_acc = tot_acc / max(1, tot_n)
    avg_loss = tot_loss / max(1, tot_n)
    return avg_acc, avg_loss
