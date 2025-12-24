# -*- coding: utf-8 -*-
"""
Loader for DermaMNIST / PathMNIST (MedMNIST)
"""

from torch.utils.data import DataLoader, Dataset
from medmnist import INFO
import medmnist

import torch

from .transforms_registry import REGISTRY


class OneHotToIndexWrapper(Dataset):
    """
    EN:
        Wrap MedMNIST dataset so that:
        - for scalar labels (e.g. [3] or 3) we just return int(label)
        - for true one-hot / multi-label vectors we use argmax.

    ZH:
        封装 MedMNIST 数据集：
        - 如果标签本身就是“单个类编号”（标量或长度 1 向量），直接 int()；
        - 只有在确实是 one-hot / 多维向量时才用 argmax。
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, target = self.base_ds[idx]

        if isinstance(target, torch.Tensor):
            arr = target
        else:
            arr = torch.as_tensor(target)

        if arr.ndim == 0 or arr.numel() == 1:
            cls = int(arr.item())
        else:
            cls = int(arr.argmax().item())

        return img, cls


def _as_list_floats(x):
    """
    EN: Convert x (list/tuple/number) into list[float].
    ZH: 把 x（list/tuple/数字）转换成 list[float]。
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    raise TypeError(f"mean/std must be number or list/tuple, got {type(x)}")


def _broadcast_mean_std(mean, std, n_channels: int):
    """
    EN:
        Make mean/std length match n_channels.
        - If mean/std is None: use 0.5 defaults
        - If length==1 and n_channels==3: replicate
        - If length==3 and n_channels==1: take first (or average)

    ZH:
        让 mean/std 的长度匹配通道数 n_channels：
        - mean/std 缺失：用 0.5 默认值
        - mean/std 只有 1 个值但需要 3 通道：复制成 3 个
        - mean/std 有 3 个值但需要 1 通道：默认取第一个（也可改成平均）
    """
    if mean is None:
        mean = [0.5] * n_channels
    if std is None:
        std = [0.5] * n_channels

    if len(mean) == 1 and n_channels == 3:
        mean = mean * 3
    if len(std) == 1 and n_channels == 3:
        std = std * 3

    if len(mean) == 3 and n_channels == 1:
        # option A: take first channel
        mean = [mean[0]]
        # option B: average
        # mean = [sum(mean)/3.0]

    if len(std) == 3 and n_channels == 1:
        std = [std[0]]
        # std = [sum(std)/3.0]

    if len(mean) != n_channels or len(std) != n_channels:
        raise ValueError(
            f"mean/std length mismatch: n_channels={n_channels}, "
            f"mean={mean}, std={std}. Please fix dataset YAML."
        )

    return tuple(mean), tuple(std)


def get_loaders(cfg):
    """
    EN:
        MedMNIST loader (DermaMNIST / PathMNIST).
        - Uses official train/val/test split from MedMNIST.
        - Builds transforms via REGISTRY with correct img_size/mean/std.

    ZH:
        MedMNIST loader（DermaMNIST / PathMNIST）：
        - 使用官方 train/val/test split
        - 用 REGISTRY 构建 transforms，并自动适配通道数的 mean/std
    """
    name = cfg["data"]["name"].lower()
    if name not in ["dermamnist", "pathmnist"]:
        raise ValueError(
            f"MedMNIST loader only supports DermaMNIST and PathMNIST, got {name}"
        )

    data_cfg = cfg["data"]
    aug_key = data_cfg["aug"].lower()
    img_size = int(data_cfg.get("img_size", 28))

    # --- dataset info (for channels/classes) ---
    info = INFO[name]
    n_channels = int(info.get("n_channels", 3))
    DataClass = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])

    # --- mean/std (robust handling) ---
    mean = _as_list_floats(data_cfg.get("mean", None))
    std  = _as_list_floats(data_cfg.get("std", None))
    mean, std = _broadcast_mean_std(mean, std, n_channels=n_channels)

    # --- transforms ---
    if aug_key not in REGISTRY:
        raise KeyError(f"Unknown augmentation key for MedMNIST: {aug_key}. "
                       f"Available: {list(REGISTRY.keys())}")

    # IMPORTANT: registry functions should accept (img_size, mean, std)
    train_tf, test_tf = REGISTRY[aug_key](img_size=img_size, mean=mean, std=std)

    # --- datasets ---
    root = data_cfg["root"]

    base_train = DataClass(root=root, split="train", transform=train_tf, download=False)
    base_val   = DataClass(root=root, split="val",   transform=test_tf,  download=False)
    base_test  = DataClass(root=root, split="test",  transform=test_tf,  download=False)

    train_set = OneHotToIndexWrapper(base_train)
    val_set   = OneHotToIndexWrapper(base_val)
    test_set  = OneHotToIndexWrapper(base_test)

    # --- loaders ---
    def dl(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=True,
        )

    train_ld = dl(train_set, True)
    val_ld   = dl(val_set,   False)
    test_ld  = dl(test_set,  False)

    return train_ld, val_ld, test_ld, num_classes