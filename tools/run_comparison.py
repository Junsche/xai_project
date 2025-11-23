"""
Stage-2: Augmentation comparison runner.

EN:
    This script runs Stage-2 experiments for one dataset.
    For the chosen dataset, it will loop over ALL augmentation configs
    under `configs/augs/*.yaml` and call `main.py` for each of them,
    using the Stage-1 baseline hyperparameters (lr, epochs).

    Usage examples:
        python tools/run_comparison.py cifar10
        python tools/run_comparison.py cifar100
        python tools/run_comparison.py dermamnist
        python tools/run_comparison.py pathmnist

ZH：
    这个脚本用于运行「阶段 2：增强对比」实验。
    对于指定的数据集，它会遍历 `configs/augs/*.yaml` 里的所有增强配置，
    并为每个增强调用一次 `main.py`，使用阶段 1 选出来的 lr 和 epochs。

    使用示例：
        python tools/run_comparison.py cifar10
        python tools/run_comparison.py cifar100
        python tools/run_comparison.py dermamnist
        python tools/run_comparison.py pathmnist
"""

import subprocess
import sys
from pathlib import Path

# ============================
# 1) 路径 & 常量 / Paths & constants
# ============================

BASE_CFG = "configs/_base-comparison.yaml"

# 每个数据集在 Stage-1 选出的超参数（可以以后根据结果微调）
# Stage-1 baseline hyperparameters per dataset
DATASETS = {
    "cifar10": {
        "yaml": "configs/datasets/cifar10.yaml",
        "lr": 0.01,
        "epochs": 100,
        "group": "cifar10",
    },
    "cifar100": {
        "yaml": "configs/datasets/cifar100.yaml",
        "lr": 0.05,
        "epochs": 100,
        "group": "cifar100",
    },
    "dermamnist": {
        "yaml": "configs/datasets/dermamnist.yaml",
        "lr": 0.0005,          # 注意：这里用 0.0005，而不是 0.001
        "epochs": 30,          # 医学数据集训练稍短一些
        "group": "dermamnist",
    },
    "pathmnist": {
        "yaml": "configs/datasets/pathmnist.yaml",
        "lr": 0.0005,
        "epochs": 30,
        "group": "pathmnist",
    },
}


def discover_augs():
    """
    自动发现所有增强配置：扫描 configs/augs 目录下的 *.yaml 文件。

    EN:
        Discover all augmentation configs automatically by scanning
        `configs/augs/*.yaml`. Returns a list of (aug_name, aug_yaml_path).
    """
    aug_dir = Path("configs/augs")
    augs = []
    for p in sorted(aug_dir.glob("*.yaml")):
        # 跳过以 "_" 开头的模板文件 / skip templates
        if p.name.startswith("_"):
            continue
        aug_name = p.stem  # e.g. "baseline", "randaugment"
        augs.append((aug_name, str(p)))
    return augs


def build_cmd(ds_name: str, ds_cfg: dict, aug_name: str, aug_yaml: str):
    """
    构造调用 main.py 的命令。
    Build the command line for calling main.py.
    """
    cmd = [
        "python",
        "main.py",
        BASE_CFG,
        ds_cfg["yaml"],
        aug_yaml,
        "--override",
        # 固定 Stage-2 的基本设置：
        # - 不再使用 train/val split，而是直接用 train+val / test
        # - 关闭早停（Stage-2 全程训练指定的 epochs）
        "data.use_val_split=false",
        "early_stopping.enabled=false",
        # W&B 项目信息 / W&B project info
        "wandb.project=aug-comparison",
        f"wandb.group={ds_cfg['group']}",
        # 使用 Stage-1 选出来的 lr 和 epochs
        # Use lr & epochs chosen in Stage-1
        f"train.lr={ds_cfg['lr']}",
        f"train.epochs={ds_cfg['epochs']}",
        # 为 Stage-2 运行单独的 exp_id
        f"train.exp_id=S2_{aug_name}",
    ]
    return cmd


def run_one_experiment(ds_name: str, ds_cfg: dict, aug_name: str, aug_yaml: str, idx: int, total: int):
    """
    运行单个数据集 + 单个增强的实验，并打印进度。
    Run one (dataset, augmentation) experiment with progress printing.
    """
    print("=" * 80)
    print(f"[{idx}/{total}] 🚀 Running Stage-2")
    print(f"Dataset : {ds_name}")
    print(f"Augment : {aug_name}")
    print(f"Config  : {aug_yaml}")
    cmd = build_cmd(ds_name, ds_cfg, aug_name, aug_yaml)
    print("Command :", " ".join(cmd))
    print("=" * 80)

    # 真正执行 / actually run
    subprocess.run(cmd, check=True)


def main():
    """
    Usage / 用法示例：

      # 跑某个数据集的所有增强
      # Run all augmentations for a dataset
      python tools/run_comparison.py cifar10

      # 只跑某一个增强（例如 cutmix）
      # Run only a single augmentation
      python tools/run_comparison.py cifar10 cutmix
    """
    # ----------------------------
    # 解析命令行参数 / Parse CLI args
    # ----------------------------
    if len(sys.argv) < 2:
        print("Usage: python tools/run_comparison.py <dataset> [aug_name]")
        print("  dataset  ∈ {cifar10, cifar100, dermamnist, pathmnist}")
        print("  aug_name ∈ YAML name in configs/augs (e.g. baseline, cutmix)")
        sys.exit(1)

    ds_name = sys.argv[1].lower()
    if ds_name not in DATASETS:
        print(f"[ERROR] Unknown dataset: {ds_name}")
        print("       Must be one of: cifar10, cifar100, dermamnist, pathmnist")
        sys.exit(1)

    # 可选：第二个参数指定只跑某个 aug
    # Optional second arg: run only a specific augmentation
    requested_aug = sys.argv[2].lower() if len(sys.argv) > 2 else None

    ds_cfg = DATASETS[ds_name]

    # ----------------------------
    # 自动发现所有增强配置
    # Discover augmentation configs
    # ----------------------------
    all_augs = discover_augs()
    if not all_augs:
        print("[ERROR] No augmentation yaml files found in configs/augs")
        sys.exit(1)

    # 打印所有可用的增强名称，方便你检查
    # Print all available augmentations
    all_aug_names = [name for name, _ in all_augs]
    print(f"\nAvailable augmentations: {', '.join(all_aug_names)}")

    # 如果用户指定了某个 aug，则只保留这一项
    # If user requested a specific aug, filter the list
    if requested_aug is not None:
        all_augs = [(name, y) for name, y in all_augs if name == requested_aug]
        if not all_augs:
            print(f"[ERROR] Aug '{requested_aug}' not found in configs/augs/")
            sys.exit(1)
        print(f"⚠️  Will run ONLY augmentation: {requested_aug}")

    total = len(all_augs)
    print(f"\nWill run Stage-2 for dataset '{ds_name}' with {total} augmentation(s).\n")

    # ----------------------------
    # 逐个增强运行实验 / Run experiments
    # ----------------------------
    for idx, (aug_name, aug_yaml) in enumerate(all_augs, start=1):
        run_one_experiment(ds_name, ds_cfg, aug_name, aug_yaml, idx, total)

    print("\n✅ All Stage-2 runs finished.\n")


if __name__ == "__main__":
    main()