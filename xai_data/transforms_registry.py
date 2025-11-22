# data/transforms_registry.py
# 作用（中文）：把图像级增强做成“注册表”，配置里用字符串选择。
# Purpose (EN): Register image-level transforms and select by key from configs.

from torchvision import transforms

# CIFAR 的标准均值/方差（Channels: RGB）
# CIFAR standard mean/std
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def baseline(img_size=32, mean=MEAN, std=STD):
    # 训练增强：RandomCrop + RandomHorizontalFlip + Normalize
    # Train augs: crop + hflip + normalize
    train_tf = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # 验证/测试：仅 Normalize
    # Val/Test: normalize only
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf

REGISTRY = {
    "baseline": baseline,
    # 未来可在此继续添加：autoaugment / randaugment / styleaug / diffusemix ...
}