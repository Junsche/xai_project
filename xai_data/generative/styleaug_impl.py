# xai_data/generative/styleaug_impl.py
# -*- coding: utf-8 -*-
"""
Lightweight StyleAug implementation (StyleAug-v0).

EN:
    This is a *simple* style-based augmentation implemented only with
    torchvision operators (no heavy pretrained GAN / diffusion).
    It performs:
        - RandomCrop + HorizontalFlip (like baseline)
        - Strong ColorJitter (brightness/contrast/saturation/hue)
        - RandomGrayscale
        - Random GaussianBlur
    and then normalizes with dataset mean/std.

ZH:
    这里是一个 *轻量* 的 StyleAug 实现，不依赖预训练 GAN / Diffusion。
    核心操作包括：
        - 与 baseline 相同的 RandomCrop + 水平翻转
        - 较强的 ColorJitter（亮度 / 对比度 / 饱和度 / 色相）
        - 随机转为灰度
        - 高斯模糊
    然后再用数据集的 mean/std 做标准化。
"""

from torchvision import transforms


def build_styleaug_transforms(
    img_size: int = 32,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
):
    """
    EN:
        Build train / test transforms for StyleAug.
        - Train: strong appearance / style perturbation
        - Test: plain ToTensor + Normalize (no augmentation)

    ZH:
        构造 StyleAug 的训练 / 测试 transform：
        - 训练：加入较强的外观 / 风格扰动
        - 测试：只做 ToTensor + Normalize，不做增强
    """

    # Color jitter with relatively strong ranges
    color_jitter = transforms.ColorJitter(
        brightness=0.4,   # ±40% brightness
        contrast=0.4,     # ±40% contrast
        saturation=0.4,   # ±40% saturation
        hue=0.1,          # ±0.1 hue
    )

    train_tf = transforms.Compose([
        # geometric baseline augs
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),

        # style-like appearance perturbations
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.5,
        ),

        # to tensor + normalize
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, test_tf