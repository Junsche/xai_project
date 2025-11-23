# 作用（中文）：把图像级增强做成“注册表”，配置里用字符串选择。
# Purpose (EN): Register image-level transforms and select by key from configs.

from torchvision import transforms
from xai_data.generative.styleaug_impl import build_styleaug_transforms
# CIFAR 的标准均值/方差（Channels: RGB）
# CIFAR standard mean/std
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)


# -------------------------
# 1) Baseline
# -------------------------
def baseline(img_size=32, mean=MEAN, std=STD):
    """
    Baseline augmentation:
    - RandomCrop + RandomHorizontalFlip + Normalize
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Val/Test: normalize only
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 2) AutoAugment
# -------------------------
def autoaugment(img_size=32, mean=MEAN, std=STD):
    """
    AutoAugment for CIFAR:
    - (optional) RandomCrop + Flip
    - AutoAugment(CIFAR10 policy)
    - Normalize
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 3) RandAugment
# -------------------------
def randaugment(img_size=32, mean=MEAN, std=STD, num_ops=2, magnitude=9):
    """
    RandAugment:
    - RandomCrop + Flip
    - RandAugment(num_ops, magnitude)
    - Normalize
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 4) Rotation + RandomErasing
# -------------------------
def rotation_erasing(img_size=32, mean=MEAN, std=STD,
                     max_deg=15, erase_p=0.5):
    """
    Traditional augmentation:
    - RandomCrop + Flip + RandomRotation
    - RandomErasing (applied on tensor)
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=max_deg),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=erase_p),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


# -------------------------
# 5) StyleAug / DiffuseMix (占位版本)
# -------------------------
def styleaug(img_size=32, mean=MEAN, std=STD):
    """
    EN:
        StyleAug-v0: geometric baseline + strong color/style perturbations.
    ZH:
        StyleAug-v0：在 baseline 的几何增强基础上，叠加颜色 / 风格扰动。
    """
    return build_styleaug_transforms(img_size=img_size, mean=mean, std=std)


def diffusemix(img_size=32, mean=MEAN, std=STD):
    """
    EN: DiffuseMix-v0: baseline geometry + blur + Gaussian noise.
    ZH: DiffuseMix-v0：基线几何增强 + 模糊 + 高斯噪声。
    """
    from xai_data.generative.diffusemix_impl import build_diffusemix_transforms
    return build_diffusemix_transforms(img_size=img_size, mean=mean, std=std)


# -------------------------
# 注册表 / Registry
# -------------------------
REGISTRY = {
    "baseline":          baseline,
    "autoaugment":       autoaugment,
    "randaugment":       randaugment,
    "rotation_erasing":  rotation_erasing,
    "styleaug":          styleaug,     # 目前 == baseline，占位
    "diffusemix":        diffusemix,   # 目前 == baseline，占位
    # 注意：Mixup / CutMix 是在 train.trainer 里通过 loss 实现，
    # 一般 data.aug 仍然用 "baseline"，由 configs/augs/*.yaml 打开 mixup_alpha / cutmix_alpha。
}