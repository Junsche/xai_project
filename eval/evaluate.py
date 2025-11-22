# eval/evaluate.py
# 中文说明：提供两个常用评估入口——(1) 干净集(eval loader)的 Accuracy/ECE；(2) CIFAR-C 的 per-corruption 错误率与 mCE。
# English: Two entry-points — (1) clean split (eval loader) Accuracy/ECE; (2) CIFAR-C per-corruption errors + mCE.

from __future__ import annotations
import os
import torch
import wandb
from typing import Dict, Tuple, Optional

from train.metrics import accuracy, expected_calibration_error
from xai_data.cifar_c import (
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
    loader,                      # torch.utils.data.DataLoader (val or test)
    device: torch.device,
    split_name: str = "val",     # "val" or "test" (for logging keys)
    do_ece: bool = True,
) -> Dict[str, float]:
    """
    中文：在给定的 DataLoader 上评估 Accuracy 和（可选）ECE，并写入 W&B。
    EN: Evaluate Accuracy (and optional ECE) on the provided loader, and log to W&B.
    """
    model.eval()
    total, correct = 0, 0
    ece_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += x.size(0)
        if do_ece:
            ece_sum += expected_calibration_error(logits, y, n_bins=15) * x.size(0)

    acc = correct / max(1, total)
    metrics = {f"{split_name}/acc": acc}

    if do_ece:
        metrics[f"{split_name}/ece"] = ece_sum / max(1, total)

    # 记录到 W&B / log to W&B
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
    dataset_tag: str = "c10",                  # "c10" or "c100" (used for ref file naming)
    use_15_set: bool = True,
    save_baseline_ref_if_missing: bool = False,
) -> Tuple[Dict[str, list], Dict[str, float], float]:
    """
    中文：在 CIFAR-C 上评估当前模型，返回 (per_severity_errors, per_corruption_avg, mCE)，并写入 W&B。
         若 normalize_by="baseline"，则尝试从 references/baseline_cifarC_{dataset_tag}.json 读取参考；
         若该文件不存在且 save_baseline_ref_if_missing=True，则会将当前模型的 per_corruption_avg 保存为参考。
    EN: Evaluate model on CIFAR-C; returns (per_sev_errs, per_corr_avg, mCE) and logs to W&B.
        If normalize_by="baseline", it loads reference from references/baseline_cifarC_{dataset_tag}.json;
        if missing and save_baseline_ref_if_missing=True, it saves current per_corr_avg as the reference.
    """
    # 选择参考 / pick normalization reference
    ref = None
    if normalize_by == "baseline":
        os.makedirs("references", exist_ok=True)
        ref_path = f"references/baseline_cifarC_{dataset_tag}.json"
        ref = load_baseline_reference(ref_path)

    per_sev, per_avg, mce = eval_model_on_cifar_c(
        model, device, cifar_c_root,
        batch_size=batch_size, num_workers=num_workers,
        normalize_ref=ref, use_15_set=use_15_set
    )

    # 逐项记录 / detailed logging
    for cname, errs in per_sev.items():
        for i, e in enumerate(errs, 1):
            wandb.log({f"cifar_c/{cname}/severity_{i}": e})
    for cname, e in per_avg.items():
        wandb.log({f"cifar_c/{cname}/avg_error": e})
    wandb.log({"cifar_c/mCE": mce})

    # 若以 baseline 归一化但参考缺失，可选择把当前曲线存为参考
    # If baseline-normalization is requested but reference is missing, optionally save current as reference
    if normalize_by == "baseline" and ref is None and save_baseline_ref_if_missing:
        save_baseline_reference(per_avg, f"references/baseline_cifarC_{dataset_tag}.json")

    return per_sev, per_avg, mce