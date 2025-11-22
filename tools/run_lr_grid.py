# tools/run_lr_grid.py
# EN: Run LR grid search (C1–C4) for a given dataset.
# ZH: 针对给定数据集，跑一组 C1–C4 学习率网格的 baseline。

import subprocess
import sys

def run(dataset: str):
    """
    EN:
      Run LR grid for one dataset.
      - CIFAR-10/100: use larger LR grid
      - DermaMNIST/PathMNIST: use smaller LR grid
    ZH：
      对单个数据集执行学习率网格搜索：
      - CIFAR-10/100：使用较大的学习率网格
      - DermaMNIST/PathMNIST：使用较小的学习率网格
    """
    dataset = dataset.lower()
    assert dataset in ["cifar10", "cifar100", "dermamnist", "pathmnist"], \
        f"Unknown dataset: {dataset}"

    ds_yaml = f"configs/datasets/{dataset}.yaml"

    # -------------------------------
    # 1) Define LR grid per dataset
    # -------------------------------
    if dataset in ["cifar10", "cifar100"]:
        # EN: CIFAR grid (your original C1–C4)
        # ZH: CIFAR 使用你原来的 C1–C4 学习率
        grid = [
            ("0.1",   "C1"),
            ("0.05",  "C2"),
            ("0.01",  "C3"),
            ("0.001", "C4"),
        ]
    else:
        # EN: MedMNIST grid (smaller LRs)
        # ZH: MedMNIST 使用更温和的学习率网格
        grid = [
            ("0.01",   "C1"),
            ("0.005",  "C2"),
            ("0.001",  "C3"),
            ("0.0005", "C4"),
        ]

    # -------------------------------
    # 2) Loop over LR candidates
    # -------------------------------
    for lr, exp in grid:
        cmd = [
            "python", "main.py",
            "configs/_base.yaml",
            ds_yaml,
            "configs/augs/baseline.yaml",
            "--override",
            f"train.lr={lr}",
            f"train.exp_id={exp}",
        ]
        print("=" * 80)
        print(f"🚀 Running {dataset} baseline: lr={lr}, exp={exp}")
        print("=" * 80)
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # EN: Use command line arg, default to cifar10
    # ZH: 命令行传入数据集名称，默认 cifar10
    ds = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    run(ds)
    
    
    
"""
# CIFAR-10 baseline (C1–C4)
python tools/run_lr_grid.py cifar10

# CIFAR-100 baseline (C1–C4)
python tools/run_lr_grid.py cifar100

# DermaMNIST baseline (C1–C4, with Balanced Accuracy)
python tools/run_lr_grid.py dermamnist

# PathMNIST baseline (C1–C4, with Balanced Accuracy)
python tools/run_lr_grid.py pathmnist
"""