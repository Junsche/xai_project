# xai_data/__init__.py
# Unified dataset dispatcher
# 统一数据加载入口，根据 cfg["data"]["name"] 自动选择调用的 loader

from .cifar import get_loaders as get_cifar_loaders

# 注意：medmnist.py 文件下一步会创建，现在先占位
try:
    from .medmnist import get_loaders as get_medmnist_loaders
except ImportError:
    get_medmnist_loaders = None


def get_loaders(cfg):
    """Unified dataset loader
    根据 cfg["data"]["name"] 自动选择加载器
    """
    name = cfg["data"]["name"].lower()

    # CIFAR 系列
    if name in ["cifar10", "cifar100"]:
        train_ld, val_ld, num_classes = get_cifar_loaders(cfg)
        return train_ld, val_ld, None, num_classes   # CIFAR 不返回 test_set（eval 时另建）

    # 医学数据集（下一步我们会创建 medmnist.get_loaders）
    if name in ["dermamnist", "pathmnist"]:
        if get_medmnist_loaders is None:
            raise ImportError("medmnist loader not implemented yet. Create xai_data/medmnist.py first.")
        train_ld, val_ld, test_set, num_classes = get_medmnist_loaders(cfg)
        return train_ld, val_ld, test_set, num_classes

    raise ValueError(f"Unknown dataset: {name}")