# -*- coding: utf-8 -*-
"""
Checkpoint naming + lookup helpers for Stage-2 / Stage-3.

This module centralizes:
- Stage-2 selected LR lookup
- augmentation token normalization
- run-name construction
- checkpoint path construction
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 1437
DEFAULT_CKPT_DIR = ROOT / "runs"
STAGE2_LR_CFG = ROOT / "configs" / "protocol" / "stage2_selected_lrs.yaml"


def _load_yaml_dict(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML is empty: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"YAML must load as dict: {path}")
    return data


def load_stage2_selected_lrs() -> Dict[str, float]:
    """
    Load the dataset -> selected LR mapping used by Stage-2.
    """
    cfg = _load_yaml_dict(STAGE2_LR_CFG)
    return cfg


def stage2_lr_for(dataset: str) -> str:
    """
    Return the Stage-2 selected LR as a string, so that checkpoint names
    exactly match the Stage-2 naming format.
    """
    dataset = str(dataset).lower()
    lr_map = load_stage2_selected_lrs()
    if dataset not in lr_map:
        raise KeyError(f"No Stage-2 selected LR found for dataset: {dataset}")
    return str(lr_map[dataset])


def normalize_aug_token(aug: str) -> str:
    """
    For checkpoint naming:
    - mixup / cutmix use baseline image-level augmentation
    - all other augmentations use their own token
    """
    aug = str(aug).lower()
    return "baseline" if aug in ["mixup", "cutmix"] else aug


def stage2_exp_id(aug: str) -> str:
    """
    Return the canonical Stage-2 exp_id for one augmentation.
    """
    aug = str(aug).lower()
    return f"S2_{aug}"


def build_stage2_run_name(
    *,
    dataset: str,
    aug: str,
    model_name: str,
    seed: int = DEFAULT_SEED,
) -> str:
    """
    Build the canonical Stage-2 run name:
    {model}_{dataset}_{aug_token}_{exp_id}_lr{lr}_seed{seed}
    """
    dataset = str(dataset).lower()
    model_name = str(model_name).lower()
    aug = str(aug).lower()

    lr = stage2_lr_for(dataset)
    aug_token = normalize_aug_token(aug)
    exp_id = stage2_exp_id(aug)

    return f"{model_name}_{dataset}_{aug_token}_{exp_id}_lr{lr}_seed{seed}"


def build_stage2_ckpt_path(
    *,
    dataset: str,
    aug: str,
    model_name: str,
    seed: int = DEFAULT_SEED,
    ckpt_dir: Path | str = DEFAULT_CKPT_DIR,
    suffix: str = "last",
) -> str:
    """
    Build the canonical Stage-2 checkpoint path.
    Example:
      runs/resnet18_cifar10_baseline_S2_baseline_lr0.01_seed1437_last.pt
    """
    ckpt_dir = Path(ckpt_dir)
    run_name = build_stage2_run_name(
        dataset=dataset,
        aug=aug,
        model_name=model_name,
        seed=seed,
    )
    return str(ckpt_dir / f"{run_name}_{suffix}.pt")


def assert_stage2_ckpt_exists(
    *,
    dataset: str,
    aug: str,
    model_name: str,
    seed: int = DEFAULT_SEED,
    ckpt_dir: Path | str = DEFAULT_CKPT_DIR,
    suffix: str = "last",
) -> str:
    """
    Return the checkpoint path if it exists, otherwise raise FileNotFoundError.
    """
    ckpt_path = build_stage2_ckpt_path(
        dataset=dataset,
        aug=aug,
        model_name=model_name,
        seed=seed,
        ckpt_dir=ckpt_dir,
        suffix=suffix,
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {ckpt_path}")
    return ckpt_path