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
    
    
    
""""
Stage-1: Baseline LR Grid Search
阶段 1：基线模型学习率网格搜索

EN:
    This script runs a learning-rate grid search for the baseline model
    on a single dataset (no advanced augmentation, only `baseline`).
    It calls `main.py` multiple times with different `train.lr` values,
    and logs all runs to W&B (typically project `aug-baseline-cnn`).

    Typical usage:
        # CIFAR datasets
        python tools/run_lr_grid.py cifar10
        python tools/run_lr_grid.py cifar100

        # Medical datasets (if supported in this script)
        python tools/run_lr_grid.py dermamnist
        python tools/run_lr_grid.py pathmnist

    The goal of Stage-1 is:
        - For EACH dataset, find a stable learning rate (and roughly a
          reasonable epoch budget) that will be reused in Stage-2.
        - We do NOT compare augmentations here, only the baseline.

ZH:
    本脚本用于对「基线模型」在单个数据集上做学习率网格搜索
    （不启用高级增强，只使用 `baseline` 增强）。
    它会用不同的 `train.lr` 多次调用 `main.py`，并把所有结果记录到
    W&B（通常是 `aug-baseline-cnn` 项目）。

    常用命令示例：
        # CIFAR 数据集
        python tools/run_lr_grid.py cifar10
        python tools/run_lr_grid.py cifar100

        # 医学数据集（如果本脚本中已支持）
        python tools/run_lr_grid.py dermamnist
        python tools/run_lr_grid.py pathmnist

    Stage-1 的目标是：
        - 对「每个数据集」分别找到一个稳定的学习率
          （以及大致合理的 epoch 数），供 Stage-2 固定使用。
        - 这一阶段不比较增强方法，只跑 baseline。
"""