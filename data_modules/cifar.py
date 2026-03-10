# Purpose: Load CIFAR-10/100 and build train/eval loaders from cfg.
# - If use_val_split=True: split train into train/val and use val as eval.
# - If use_val_split=False: use official test set as eval (Stage-2 behavior).

from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from .transforms_registry import REGISTRY


def _get_aug_params(data_cfg: dict, aug_key: str) -> dict:
   
    block = data_cfg.get(aug_key, {})
    return block if isinstance(block, dict) else {}


def get_loaders(cfg):
    data_cfg = cfg["data"]
    aug_key = str(data_cfg["aug"]).lower()

    # --- Safety: validate augmentation key ---
    if aug_key not in REGISTRY:
        raise KeyError(
            f"Unknown augmentation key for CIFAR: {aug_key}. "
            f"Available: {list(REGISTRY.keys())}"
        )

    # --- Read shared transform args from dataset cfg ---
    # These keep default values if not provided (backward compatible).
    img_size = int(data_cfg.get("img_size", 32))
    mean = tuple(data_cfg.get("mean", (0.4914, 0.4822, 0.4465)))
    std  = tuple(data_cfg.get("std",  (0.2470, 0.2435, 0.2616)))

    # --- Optional per-augmentation parameters (empty by default) ---
    aug_params = _get_aug_params(data_cfg, aug_key)

    # Build transforms: pass shared args + optional aug-specific params
    train_tf, test_tf = REGISTRY[aug_key](
        img_size=img_size,
        mean=mean,
        std=std,
        **aug_params
    )

    name = str(data_cfg["name"]).lower()

    # --- Build datasets ---
    if name == "cifar10":
        full_train = datasets.CIFAR10(
            data_cfg["root"], train=True, download=False, transform=train_tf
        )
        test_set = datasets.CIFAR10(
            data_cfg["root"], train=False, download=False, transform=test_tf
        )
        num_classes = 10

    elif name == "cifar100":
        full_train = datasets.CIFAR100(
            data_cfg["root"], train=True, download=False, transform=train_tf
        )
        test_set = datasets.CIFAR100(
            data_cfg["root"], train=False, download=False, transform=test_tf
        )
        num_classes = 100

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # --- Split train/val or use test as eval set ---
    if bool(data_cfg.get("use_val_split", False)):
        val_ratio = float(data_cfg.get("val_split_ratio", 0.1))
        val_len = int(len(full_train) * val_ratio)
        train_len = len(full_train) - val_len
        train_set, val_set = random_split(full_train, [train_len, val_len])
        eval_set = val_set
    else:
        train_set = full_train
        eval_set = test_set

    # --- DataLoaders ---
    def dl(ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=True,
        )

    train_ld = dl(train_set, True)
    eval_ld  = dl(eval_set, False)

    return train_ld, eval_ld, num_classes
