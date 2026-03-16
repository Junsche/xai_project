# -*- coding: utf-8 -*-

from typing import Callable

import torch
from torchvision import transforms


def _add_gaussian_noise(std: float = 0.05) -> Callable:
    """
    EN:
    Add Gaussian noise after ToTensor().

    ZH:
    在 ToTensor() 之后加入高斯噪声。
    """
    def _fn(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        noise = torch.randn_like(x) * std
        x = x + noise
        return x.clamp_(0.0, 1.0)

    return transforms.Lambda(_fn)


def build_diffusemix_core():
    """
    EN:
    Build the core DiffuseMix-like perturbation block.
    No resize / crop / flip is included here.

    ZH:
    构造 DiffuseMix 风格的核心扰动模块。
    这里不包含 resize / crop / flip。
    """
    return transforms.Compose([
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.7,
        ),
    ])