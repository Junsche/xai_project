# eval/evaluate.py
# 中文说明：提供两个常用评估入口——
# (1) 干净集(eval loader)的 Accuracy/ECE；
# (2) CIFAR-C 的 per-corruption 错误率与 mCE。
# English: Two entry-points —
# (1) clean split (eval loader) Accuracy/ECE;
# (2) CIFAR-C per-corruption errors + mCE.

from __future__ import annotations
import os
import torch
import wandb
from typing import Dict, Tuple, Optional

from train.metrics import accuracy, expected_calibration_error
from data_modules.cifar_c import (
    eval_model_on_cifar_c,
    load_baseline_reference,
    save_baseline_reference,
)

# -----------------------------
# (A) Clean eval: Accuracy / ECE
# -----------------------------
@torch.no_grad()
def eval_clean_and_log(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    split_name: str = "val",
    do_ece: bool = True,
) -> Dict[str, float]:
    
    model.eval()
    total, correct = 0, 0
    ece_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        if do_ece:
            ece_sum += expected_calibration_error(logits, y, n_bins=15) * x.size(0)

    acc = correct / max(1, total)
    metrics = {f"{split_name}/acc": acc}

    if do_ece:
        metrics[f"{split_name}/ece"] = ece_sum / max(1, total)

    wandb.log(metrics)
    return metrics


# ----------------------------------------
# (B) CIFAR-C: per-corruption errors & mCE
# ----------------------------------------
@torch.no_grad()
def eval_cifar_c_and_log(
    model: torch.nn.Module,
    device: torch.device,
    cifar_c_root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    normalize_by: Optional[str] = "baseline",  # "baseline" | "alexnet" | None
    dataset_tag: str = "c10",                  # "c10" or "c100"
    use_15_set: bool = True,
    save_baseline_ref_if_missing: bool = False,
) -> Tuple[Dict[str, list], Dict[str, float], float]:
    
    ref = None
    if normalize_by == "baseline":
        os.makedirs("references", exist_ok=True)
        ref_path = f"references/baseline_cifarC_{dataset_tag}.json"
        ref = load_baseline_reference(ref_path)

    per_sev, per_avg, mce = eval_model_on_cifar_c(
        model,
        device,
        cifar_c_root,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_ref=ref,
        use_15_set=use_15_set,
    )

    for cname, errs in per_sev.items():
        for i, e in enumerate(errs, 1):
            wandb.log({f"cifar_c/{cname}/severity_{i}": e})

    for cname, e in per_avg.items():
        wandb.log({f"cifar_c/{cname}/avg_error": e})

    wandb.log({"cifar_c/mCE": mce})

    if normalize_by == "baseline" and ref is None and save_baseline_ref_if_missing:
        save_baseline_reference(per_avg, f"references/baseline_cifarC_{dataset_tag}.json")

    return per_sev, per_avg, mce