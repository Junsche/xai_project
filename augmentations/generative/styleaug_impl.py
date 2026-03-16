# -*- coding: utf-8 -*-
"""
Lightweight StyleAug core implementation.

EN:
This module provides only the core style perturbation block.
Spatial preprocessing (resize / crop / flip) is handled centrally
by transforms_registry.py.

ZH:
该模块只提供 StyleAug 的核心风格扰动模块。
空间预处理（resize / crop / flip）统一由 transforms_registry.py 负责。
"""

from torchvision import transforms


def build_styleaug_core():
    """
    EN:
    Build the core StyleAug transform block.
    No resize / crop / flip is included here.

    ZH:
    构造 StyleAug 的核心增强模块。
    这里不包含 resize / crop / flip。
    """
    color_jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    )

    return transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.5,
        ),
    ])