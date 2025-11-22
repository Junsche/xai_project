# -*- coding: utf-8 -*-
"""
Loader for DermaMNIST / PathMNIST
"""

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import INFO
import medmnist
import torch

from .transforms_registry import REGISTRY


# -------------------------------------------------
# 小包装：把 one-hot / 多维 label 变成整数类别
# -------------------------------------------------
class OneHotToIndexWrapper(Dataset):
    """
    把 MedMNIST 返回的 one-hot / 多维 label 转成整数类别下标。
    EN: Convert MedMNIST one-hot labels to integer class indices.
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, target = self.base_ds[idx]  # target: e.g. [0,0,1,0,...] 或 numpy 向量

        # MedMNIST 的 target 通常是多维向量 (one-hot)，这里转成单个 int
        if hasattr(target, "argmax"):
            cls = int(target.argmax())
        else:
            # 保险：如果已经是标量就直接转成 int
            cls = int(target)

        return img, cls


# -------------------------------------------------
# 构建 DataLoader
# -------------------------------------------------
def get_loaders(cfg):
    """
    与 cifar.get_loaders 功能一致：
    - 支持 train/val/test
    - 医学数据集直接使用官方提供的 train / val / test
    - 使用注册表里的 augmentation
    """

    name = cfg["data"]["name"].lower()
    if name not in ["dermamnist", "pathmnist"]:
        raise ValueError(
            f"MedMNIST loader only supports DermaMNIST and PathMNIST, got {name}"
        )

    # ------------------------------
    # transforms
    # ------------------------------
    train_tf, test_tf = REGISTRY[cfg["data"]["aug"]]()

    # ------------------------------
    # dataset info
    # ------------------------------
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])

    # MedMNIST 的 INFO 没有 n_classes 字段，用 label 的长度来确定类别数
    # MedMNIST INFO has no `n_classes`; infer it from the number of labels.
    num_classes = len(info["label"])

    root = cfg["data"]["root"]

    # ------------------------------
    # 官方提供 train / val / test split
    # 先构造原始数据集，再用 OneHotToIndexWrapper 包一层
    # ------------------------------
    base_train = DataClass(
        root=root, split="train", transform=train_tf, download=False
    )
    base_val = DataClass(
        root=root, split="val", transform=test_tf, download=False
    )
    base_test = DataClass(
        root=root, split="test", transform=test_tf, download=False
    )

    train_set = OneHotToIndexWrapper(base_train)
    val_set = OneHotToIndexWrapper(base_val)
    test_set = OneHotToIndexWrapper(base_test)

    # ------------------------------
    # DataLoader
    # ------------------------------
    def dl(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=shuffle,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )

    train_ld = dl(train_set, True)
    val_ld = dl(val_set, False)
    test_ld = dl(test_set, False)

    return train_ld, val_ld, test_ld, num_classes