# tools/run_lr_grid.py
# Run Stage-1 baseline learning-rate grid search (C1–C4) for a given dataset.

import subprocess
import sys


def run(dataset: str) -> None:
    """
    Run LR grid for one dataset.

    - CIFAR-10/100 uses a larger LR grid (0.1, 0.05, 0.01, 0.001)
    - DermaMNIST/PathMNIST uses a smaller LR grid (0.01, 0.005, 0.001, 0.0005)

    Each run calls:
        python main.py configs/_base.yaml <dataset_yaml> configs/augs/baseline.yaml --override ...
    """
    dataset = dataset.lower()
    assert dataset in ["cifar10", "cifar100", "dermamnist", "pathmnist"], \
        f"Unknown dataset: {dataset}"

    ds_yaml = f"configs/datasets/{dataset}.yaml"

    # 1) Define LR grid per dataset
    if dataset in ["cifar10", "cifar100"]:
        grid = [
            ("0.1",   "C1"),
            ("0.05",  "C2"),
            ("0.01",  "C3"),
            ("0.001", "C4"),
        ]
    else:
        grid = [
            ("0.01",   "C1"),
            ("0.005",  "C2"),
            ("0.001",  "C3"),
            ("0.0005", "C4"),
        ]

    # 2) Loop over LR candidates
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
        print(f"[Stage-1] Running {dataset} baseline: lr={lr}, exp={exp}")
        print("=" * 80)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "cifar10"
    run(ds)
