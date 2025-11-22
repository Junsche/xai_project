# data/cifar.py
# 作用（中文）：按 cfg 加载 CIFAR-10/100；支持 use_val_split 自动划分训练/验证集。
# Purpose (EN): Load CIFAR-10/100; optionally split train into train/val.

from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from .transforms_registry import REGISTRY

def get_loaders(cfg):
    # 选择增强方案 / pick transforms
    train_tf, test_tf = REGISTRY[cfg["data"]["aug"]]()
    name = cfg["data"]["name"].lower()

    # 构建原始 Dataset / build datasets
    if name == "cifar10":
        full_train = datasets.CIFAR10(cfg["data"]["root"], train=True,  download=False, transform=train_tf)
        test_set   = datasets.CIFAR10(cfg["data"]["root"], train=False, download=False, transform=test_tf)
        num_classes = 10
    elif name == "cifar100":
        full_train = datasets.CIFAR100(cfg["data"]["root"], train=True,  download=False, transform=train_tf)
        test_set   = datasets.CIFAR100(cfg["data"]["root"], train=False, download=False, transform=test_tf)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # 划分 train/val 或使用 test 作为评估集
    # split train/val or use test as eval set
    if cfg["data"]["use_val_split"]:
        val_len = int(len(full_train) * cfg["data"]["val_split_ratio"])
        train_len = len(full_train) - val_len
        train_set, val_set = random_split(full_train, [train_len, val_len])
        eval_set = val_set
    else:
        train_set = full_train
        eval_set = test_set

    # DataLoader
    dl = lambda ds, sh: DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=sh,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True
    )
    train_ld = dl(train_set, True)
    eval_ld  = dl(eval_set, False)

    return train_ld, eval_ld, num_classes