# xai_data/transforms_registry.py
# Purpose: Register image-level transforms and select by key from configs.

from __future__ import annotations
from typing import Sequence, Tuple, Union, Optional

from torchvision import transforms

# Default CIFAR mean/std (RGB)
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

MeanStd = Union[Sequence[float], Tuple[float, ...]]


def _norm(mean: MeanStd, std: MeanStd) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def _default_padding(img_size: int) -> int:
    # CIFAR traditionally uses padding=4 for 32x32 (≈ 12.5%).
    # For other sizes, scale padding roughly the same.
    return max(1, int(round(img_size * 0.125)))


# -------------------------
# 1) Baseline
# -------------------------
def baseline(img_size: int = 32, mean: MeanStd = MEAN, std: MeanStd = STD):
    pad = _default_padding(img_size)
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 2) AutoAugment
# -------------------------
def autoaugment(img_size: int = 32, mean: MeanStd = MEAN, std: MeanStd = STD):
    pad = _default_padding(img_size)
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 3) RandAugment
# -------------------------
def randaugment(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    num_ops: int = 2,
    magnitude: int = 9,
):
    pad = _default_padding(img_size)
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 4) Rotation + RandomErasing
# -------------------------
def rotation_erasing(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    max_deg: int = 15,
    erase_p: float = 0.5,
):
    pad = _default_padding(img_size)
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=max_deg),
        transforms.ToTensor(),
        _norm(mean, std),
        transforms.RandomErasing(p=erase_p),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 5) StyleAug / DiffuseMix
# -------------------------
def styleaug(img_size: int = 32, mean: MeanStd = MEAN, std: MeanStd = STD):
    # Implemented in xai_data/generative/styleaug_impl.py
    from xai_data.generative.styleaug_impl import build_styleaug_transforms
    return build_styleaug_transforms(img_size=img_size, mean=mean, std=std)


def diffusemix(img_size: int = 32, mean: MeanStd = MEAN, std: MeanStd = STD):
    # Implemented in xai_data/generative/diffusemix_impl.py
    from xai_data.generative.diffusemix_impl import build_diffusemix_transforms
    return build_diffusemix_transforms(img_size=img_size, mean=mean, std=std)


# -------------------------
# 6) AugMix (generic)
# -------------------------
def augmix(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
):
    pad = _default_padding(img_size)
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.AugMix(
            severity=severity,
            mixture_width=width,
            chain_depth=depth,
            alpha=alpha,
        ),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# Registry
# -------------------------
REGISTRY = {
    "baseline": baseline,
    "autoaugment": autoaugment,
    "randaugment": randaugment,
    "rotation_erasing": rotation_erasing,
    "styleaug": styleaug,
    "diffusemix": diffusemix,
    "augmix": augmix,
    # NOTE: Mixup/CutMix are batch-level mixing in train.trainer
    # enabled via YAML (mixup_alpha / cutmix_alpha).
}