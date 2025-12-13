# -*- coding: utf-8 -*-
"""
Loader for DermaMNIST / PathMNIST
"""

from torch.utils.data import DataLoader, Dataset
from medmnist import INFO
import medmnist

from .transforms_registry import REGISTRY


from torch.utils.data import Dataset
import torch
import numpy as np

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

        # 统一转成 numpy / tensor，方便判断形状
        if isinstance(target, torch.Tensor):
            arr = target
        else:
            arr = torch.as_tensor(target)

        # 情况 1：标量或长度为 1 → 直接取数值
        # Case 1: scalar or length-1 → just take its value
        if arr.ndim == 0 or arr.numel() == 1:
            cls = int(arr.item())
        else:
            # 情况 2：真正的 one-hot / 多维标签 → 用 argmax
            # Case 2: real one-hot / multi-dim label → use argmax
            cls = int(arr.argmax().item())

        return img, cls
    

def get_loaders(cfg):
    """
    MedMNIST loader (DermaMNIST / PathMNIST).

    - Uses official train/val/test split from MedMNIST.
    - Uses the same augmentation registry as CIFAR, but
      with dataset-specific img_size / mean / std coming
      from the dataset YAML.
    """
    name = cfg["data"]["name"].lower()
    if name not in ["dermamnist", "pathmnist"]:
        raise ValueError(
            f"MedMNIST loader only supports DermaMNIST and PathMNIST, got {name}"
        )

    data_cfg = cfg["data"]
    aug_key = data_cfg["aug"]           # e.g. "baseline"
    img_size = int(data_cfg.get("img_size", 28))
    mean = tuple(data_cfg.get("mean", [0.5, 0.5, 0.5]))
    std  = tuple(data_cfg.get("std",  [0.5, 0.5, 0.5]))

    # ------------------------------------------------
    # 1) build transforms using registry + MedMNIST cfg
    # ------------------------------------------------
    if aug_key not in REGISTRY:
        raise KeyError(f"Unknown augmentation key for MedMNIST: {aug_key}")

    train_tf, test_tf = REGISTRY[aug_key](img_size=img_size, mean=mean, std=std)

    # ------------------------------------------------
    # 2) dataset info + official splits
    # ------------------------------------------------
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])

    root = data_cfg["root"]

    base_train = DataClass(root=root, split="train",
                           transform=train_tf, download=False)
    base_val   = DataClass(root=root, split="val",
                           transform=test_tf, download=False)
    base_test  = DataClass(root=root, split="test",
                           transform=test_tf, download=False)

    train_set = OneHotToIndexWrapper(base_train)
    val_set   = OneHotToIndexWrapper(base_val)
    test_set  = OneHotToIndexWrapper(base_test)

    # ------------------------------------------------
    # 3) DataLoaders
    # ------------------------------------------------
    def dl(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=data_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
        )

    train_ld = dl(train_set, True)
    val_ld   = dl(val_set,   False)
    test_ld  = dl(test_set,  False)

    return train_ld, val_ld, test_ld, num_classes