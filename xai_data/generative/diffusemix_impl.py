# -*- coding: utf-8 -*-
"""
Lightweight DiffuseMix implementation (DiffuseMix-v0).

EN:
    This is a simple diffusion-style augmentation, NOT the full DiffuseMix paper.
    Pipeline:
        - RandomCrop + HorizontalFlip (same as baseline)
        - Random GaussianBlur
        - Add Gaussian noise in tensor space
        - Normalize with dataset mean/std

ZH:
    这是一个“扩散风格”的轻量增强（不是论文中的完整 DiffuseMix）。
    处理流程：
        - 与 baseline 相同的 RandomCrop + 水平翻转
        - 随机高斯模糊
        - 在 tensor 空间添加高斯噪声
        - 使用数据集 mean/std 进行标准化
"""

from typing import Tuple

import torch
from torchvision import transforms


def _add_gaussian_noise(std: float = 0.05):
    """
    EN: Return a transform that adds N(0, std^2) noise to a tensor.
    ZH: 返回一个在 tensor 上加 N(0, std^2) 噪声的 transform。
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        noise = torch.randn_like(x) * std
        x = x + noise
        return x.clamp_(0.0, 1.0)
    return transforms.Lambda(_fn)


def build_diffusemix_transforms(
    img_size: int = 32,
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
):
    """
    EN:
        Build train / test transforms for DiffuseMix-v0.

        Train:
            - RandomCrop + HorizontalFlip
            - Random GaussianBlur
            - Add Gaussian noise
            - ToTensor + Normalize

        Test:
            - ToTensor + Normalize (no augmentation)

    ZH:
        构造 DiffuseMix-v0 的训练 / 测试 transform。

        训练：
            - RandomCrop + 水平翻转
            - 随机高斯模糊
            - 添加高斯噪声
            - ToTensor + Normalize

        测试：
            - 只做 ToTensor + Normalize，不做增强
    """

    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),

        # diffusion-style appearance perturbations
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.7,
        ),

        transforms.ToTensor(),
        _add_gaussian_noise(std=0.05),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, test_tf