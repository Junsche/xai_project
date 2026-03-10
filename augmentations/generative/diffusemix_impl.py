# -*- coding: utf-8 -*-


from typing import Tuple

import torch
from torchvision import transforms


def _add_gaussian_noise(std: float = 0.05):
   
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