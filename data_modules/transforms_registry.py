# Purpose: Register image-level transforms and select by key from configs.
# Supports multiple model preprocessing styles, e.g. CIFAR-style and ImageNet-style.

from __future__ import annotations
from typing import Sequence, Tuple, Union

from torchvision import transforms

# Default CIFAR mean/std (RGB)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

MeanStd = Union[Sequence[float], Tuple[float, ...]]


def _norm(mean: MeanStd, std: MeanStd) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def _default_padding(img_size: int) -> int:
    # CIFAR traditionally uses padding=4 for 32x32 (≈12.5%).
    # For other sizes, scale padding roughly the same.
    return max(1, int(round(img_size * 0.125)))


def _train_spatial_transforms(
    img_size: int,
    preprocessing: str = "cifar",
):
    """
    Build model-aware train-time spatial transforms.

    preprocessing:
      - "cifar": small-image style preprocessing
      - "imagenet": ImageNet / ViT-style preprocessing
    """
    preprocessing = str(preprocessing).lower()

    if preprocessing == "cifar":
        pad = _default_padding(img_size)
        return [
            transforms.RandomCrop(img_size, padding=pad),
            transforms.RandomHorizontalFlip(),
        ]

    if preprocessing == "imagenet":
        resize_size = max(img_size, int(round(img_size * 256 / 224)))
        return [
            transforms.Resize(resize_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ]

    raise ValueError(f"Unknown preprocessing mode: {preprocessing}")


def _test_spatial_transforms(
    img_size: int,
    preprocessing: str = "cifar",
):
    """
    Build model-aware test-time spatial transforms.
    """
    preprocessing = str(preprocessing).lower()

    if preprocessing == "cifar":
        return []

    if preprocessing == "imagenet":
        resize_size = max(img_size, int(round(img_size * 256 / 224)))
        return [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
        ]

    raise ValueError(f"Unknown preprocessing mode: {preprocessing}")


# -------------------------
# 1) Baseline
# -------------------------
def baseline(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    preprocessing: str = "cifar",
):
    train_tf = transforms.Compose([
        *_train_spatial_transforms(img_size, preprocessing),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    test_tf = transforms.Compose([
        *_test_spatial_transforms(img_size, preprocessing),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 2) AutoAugment
# -------------------------
def autoaugment(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    preprocessing: str = "cifar",
):
    policy = (
        transforms.AutoAugmentPolicy.CIFAR10
        if str(preprocessing).lower() == "cifar"
        else transforms.AutoAugmentPolicy.IMAGENET
    )

    train_tf = transforms.Compose([
        *_train_spatial_transforms(img_size, preprocessing),
        transforms.AutoAugment(policy=policy),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    test_tf = transforms.Compose([
        *_test_spatial_transforms(img_size, preprocessing),
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
    preprocessing: str = "cifar",
    num_ops: int = 2,
    magnitude: int = 9,
):
    train_tf = transforms.Compose([
        *_train_spatial_transforms(img_size, preprocessing),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    test_tf = transforms.Compose([
        *_test_spatial_transforms(img_size, preprocessing),
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
    preprocessing: str = "cifar",
    max_deg: int = 15,
    erase_p: float = 0.5,
):
    train_tf = transforms.Compose([
        *_train_spatial_transforms(img_size, preprocessing),
        transforms.RandomRotation(degrees=max_deg),
        transforms.ToTensor(),
        _norm(mean, std),
        transforms.RandomErasing(p=erase_p),
    ])
    test_tf = transforms.Compose([
        *_test_spatial_transforms(img_size, preprocessing),
        transforms.ToTensor(),
        _norm(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 5) StyleAug / DiffuseMix
# -------------------------
def styleaug(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    preprocessing: str = "cifar",
):
    """
    EN:
    StyleAug with centralized spatial preprocessing.
    This guarantees compatibility with both CIFAR-style CNN pipelines
    and ImageNet-style ViT pipelines.

    ZH:
    使用统一空间预处理的 StyleAug。
    这样可以同时兼容 CIFAR 风格的 CNN 路径和 ImageNet 风格的 ViT 路径。
    """
    from augmentations.generative import build_styleaug_core

    train_tf = transforms.Compose([
        *_train_spatial_transforms(img_size, preprocessing),
        build_styleaug_core(),
        transforms.ToTensor(),
        _norm(mean, std),
    ])

    test_tf = transforms.Compose([
        *_test_spatial_transforms(img_size, preprocessing),
        transforms.ToTensor(),
        _norm(mean, std),
    ])

    return train_tf, test_tf


def diffusemix(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    preprocessing: str = "cifar",
):
    """
    EN:
    DiffuseMix with centralized spatial preprocessing.
    This guarantees compatibility with both CIFAR-style CNN pipelines
    and ImageNet-style ViT pipelines.

    ZH:
    使用统一空间预处理的 DiffuseMix。
    这样可以同时兼容 CIFAR 风格的 CNN 路径和 ImageNet 风格的 ViT 路径。
    """
    from augmentations.generative import build_diffusemix_core, _add_gaussian_noise

    train_tf = transforms.Compose([
        *_train_spatial_transforms(img_size, preprocessing),
        build_diffusemix_core(),
        transforms.ToTensor(),
        _add_gaussian_noise(std=0.05),
        _norm(mean, std),
    ])

    test_tf = transforms.Compose([
        *_test_spatial_transforms(img_size, preprocessing),
        transforms.ToTensor(),
        _norm(mean, std),
    ])

    return train_tf, test_tf


# -------------------------
# 6) AugMix
# -------------------------
def augmix(
    img_size: int = 32,
    mean: MeanStd = MEAN,
    std: MeanStd = STD,
    preprocessing: str = "cifar",
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
):
    train_tf = transforms.Compose([
        *_train_spatial_transforms(img_size, preprocessing),
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
        *_test_spatial_transforms(img_size, preprocessing),
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