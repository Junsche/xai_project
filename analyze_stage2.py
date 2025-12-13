# analyze_stage2.py
# EN: Analyze Stage-2 clean results for CIFAR + MedMNIST.
# ZH: 对 Stage-2（干净数据集）的 CIFAR + MedMNIST 结果做汇总 & 画图。

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 固定 augmentation 排序（对应：传统增强 → 自动增强 → 图像混合 → 风格与生成式）
AUG_ORDER = [
    "baseline",
    "rotation_erasing",
    "autoaugment",
    "randaugment",
    "mixup",
    "cutmix",
    "styleaug",
    "diffusemix",
]


def summarize_dataset(df, dataset: str, out_dir: str):
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        print(f"[WARN] No runs for dataset={dataset} in CSV.")
        return

    # -------- 统一 augmentation 排序 + 好看名字 --------
    AUG_ORDER = [
        "baseline",
        "rotation_erasing",
        "autoaugment",
        "randaugment",
        "mixup",
        "cutmix",
        "styleaug",
        "diffusemix",
    ]
    AUG_NAME_MAP = {
        "baseline": "baseline",
        "rotation_erasing": "rotation erasing",
        "autoaugment": "AutoAugment",
        "randaugment": "RandAugment",
        "mixup": "Mixup",
        "cutmix": "CutMix",
        "styleaug": "StyleAug",
        "diffusemix": "DiffuseMix",
    }

    # 只保留我们关心的 augmentations（如果有的话）
    sub = sub[sub["augmentation"].isin(AUG_ORDER)].copy()

    # 聚合：每个 aug 一行
    agg = (
        sub.groupby("augmentation", dropna=False)[
            ["eval/acc", "eval/bal_acc", "eval/ece", "eval/mce", "eval/loss"]
        ]
        .mean()
    )

    # 按固定顺序重排
    agg = agg.reindex(AUG_ORDER, copy=False)
    agg.index = [AUG_NAME_MAP.get(a, a) for a in agg.index]

    # -------- 导出 LaTeX 表（单行表头 + 下划线自动转义） --------
    os.makedirs(out_dir, exist_ok=True)

    table = agg.reset_index().rename(
        columns={
            "augmentation": "Augmentation",
            "eval/acc": "Accuracy",
            "eval/bal_acc": "Balanced Acc.",
            "eval/ece": "ECE",
            "eval/mce": "MCE",
            "eval/loss": "Loss",
        }
    )

    csv_path = os.path.join(out_dir, f"stage2_summary_{dataset}.csv")
    table.to_csv(csv_path, index=False)
    print(f"[INFO] Saved summary CSV: {csv_path}")

    tex_path = os.path.join(out_dir, f"stage2_summary_{dataset}.tex")
    # escape=True：自动把 _ 等特殊字符转成 LaTeX 安全写法
    table.to_latex(
        tex_path,
        index=False,
        float_format=lambda x: f"{x:.4f}",
        escape=True,
        caption=f"Stage-2 clean results on {dataset.capitalize()}.",
        label=f"tab:stage2_{dataset}",
    )
    print(f"[INFO] Saved LaTeX table: {tex_path}")

    # -------- 画图部分可以保持不变，只是用 agg --------
    sns.set(style="whitegrid")

    def barplot_metric(metric, ylabel):
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(
            data=agg.reset_index().rename(columns={"index": "Augmentation"}),
            x=agg.index,
            y=metric,
        )
        ax.set_title(f"{dataset} - {metric}")
        ax.set_xlabel("Augmentation")
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        png_path = os.path.join(
            out_dir, f"{dataset}_{metric.replace('/', '_')}.png"
        )
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved plot: {png_path}")

    barplot_metric("eval/acc", "Accuracy")
    if agg["eval/bal_acc"].notna().any():
        barplot_metric("eval/bal_acc", "Balanced Accuracy")
    if agg["eval/ece"].notna().any():
        barplot_metric("eval/ece", "ECE")
    if agg["eval/mce"].notna().any():
        barplot_metric("eval/mce", "MCE")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Stage-2 clean runs for CIFAR + MedMNIST."
    )
    parser.add_argument(
        "--csv",
        default="analysis/stage2_all_runs.csv",
        help="Path to Stage-2 CSV exported by export_wandb_runs.py",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.csv}")

    out_dir = "analysis/plots_stage2"
    for ds in ["cifar10", "cifar100", "dermamnist", "pathmnist"]:
        summarize_dataset(df, ds, out_dir)


if __name__ == "__main__":
    main()