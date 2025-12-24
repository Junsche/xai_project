# -*- coding: utf-8 -*-
"""
Analyze Stage-3 results (clean + CIFAR-C) exported from W&B into a CSV.
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 固定 augmentation 排序（和 Stage-2 保持一致）
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


def _ensure_corruption_severity(df: pd.DataFrame) -> pd.DataFrame:
    if "corruption" in df.columns:
        corr = df["corruption"]
    elif "meta/corruption" in df.columns:
        corr = df["meta/corruption"]
    else:
        corr = pd.Series(["clean"] * len(df), index=df.index)
    df["corruption"] = corr

    if "severity" in df.columns:
        sev = df["severity"]
    elif "meta/severity" in df.columns:
        sev = df["meta/severity"]
    else:
        sev = pd.Series([0] * len(df), index=df.index)
    df["severity"] = sev
    return df


def analyze(csv_path: str, dataset: str, out_dir: str = "analysis/plots"):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    df = df[df["dataset"] == dataset]
    if df.empty:
        print(f"[WARN] No rows for dataset={dataset} in {csv_path}")
        return

    if "state" in df.columns:
        df = df[df["state"] == "finished"].copy()

    df = _ensure_corruption_severity(df)

    clean_mask = (df["corruption"] == "clean") | (df["severity"] == 0)
    df_clean = df[clean_mask].copy()
    df_c = df[~clean_mask].copy()

    if "augmentation" not in df.columns:
        raise ValueError("Column 'augmentation' not found in CSV.")
    aug_col = "augmentation"

    # ==========================
    # A) Clean test summary
    # ==========================
    metric_cols = [c for c in ["eval/acc", "eval/ece", "eval/mce", "eval/loss"]
                   if c in df_clean.columns]

    clean_summary = df_clean.groupby(aug_col)[metric_cols].mean()

    # 使用统一 augmentation 顺序
    ordered_augs = [a for a in AUG_ORDER if a in clean_summary.index]
    clean_summary = clean_summary.reindex(ordered_augs)

    print("\n=== Clean Test Summary (dataset={}) ===".format(dataset))
    print(clean_summary)

    if "eval/acc" in clean_summary.columns:
        plt.figure(figsize=(8, 5))
        plt.bar(clean_summary.index, clean_summary["eval/acc"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset}: Clean Test Accuracy per Augmentation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_clean_accuracy.png"))
        plt.close()

    if "eval/ece" in clean_summary.columns:
        plt.figure(figsize=(8, 5))
        plt.bar(clean_summary.index, clean_summary["eval/ece"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("ECE")
        plt.title(f"{dataset}: Clean Test ECE per Augmentation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_clean_ece.png"))
        plt.close()

    # ==========================
    # B) CIFAR-C mean metrics
    # ==========================
    if df_c.empty:
        print("[WARN] No CIFAR-C rows found, skip corruption analysis.")
        return

    mean_acc = df_c.groupby(aug_col)["eval/acc"].mean() if "eval/acc" in df_c.columns else None
    mean_ece = df_c.groupby(aug_col)["eval/ece"].mean() if "eval/ece" in df_c.columns else None
    mean_mce = df_c.groupby(aug_col)["eval/mce"].mean() if "eval/mce" in df_c.columns else None
    mean_loss = df_c.groupby(aug_col)["eval/loss"].mean() if "eval/loss" in df_c.columns else None

    summary_c = pd.DataFrame({
        "mean_acc": mean_acc,
        "mean_ece": mean_ece,
        "mean_mce": mean_mce,
        "mean_loss": mean_loss,
    })

    # 同样用统一的 augmentation 顺序
    summary_c = summary_c.reindex(ordered_augs)

    print("\n=== CIFAR-C Mean Metrics (dataset={}) ===".format(dataset))
    print(summary_c)

    if mean_acc is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(summary_c.index, summary_c["mean_acc"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset}: Mean Corruption Accuracy per Augmentation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_mean_c_acc.png"))
        plt.close()

    if mean_ece is not None:
        plt.figure(figsize=(8, 5))
        plt.bar(summary_c.index, summary_c["mean_ece"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("ECE")
        plt.title(f"{dataset}: Mean Corruption ECE per Augmentation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_mean_c_ece.png"))
        plt.close()

    # ==========================
    # C) Severity-wise mean curves
    # ==========================
    if "eval/acc" in df_c.columns:
        sev_mean_acc = df_c.groupby("severity")["eval/acc"].mean()
        plt.figure(figsize=(6, 4))
        plt.plot(sev_mean_acc.index, sev_mean_acc.values, marker="o")
        plt.xlabel("Severity")
        plt.ylabel("Mean Accuracy")
        plt.title(f"{dataset}: Severity-wise Mean Accuracy (all augs)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_severity_curve_acc.png"))
        plt.close()

    if "eval/ece" in df_c.columns:
        sev_mean_ece = df_c.groupby("severity")["eval/ece"].mean()
        plt.figure(figsize=(6, 4))
        plt.plot(sev_mean_ece.index, sev_mean_ece.values, marker="o")
        plt.xlabel("Severity")
        plt.ylabel("Mean ECE")
        plt.title(f"{dataset}: Severity-wise Mean ECE (all augs)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_severity_curve_ece.png"))
        plt.close()

    # ==========================
    # D) Corruption × Aug heatmap
    # ==========================
    if "eval/acc" in df_c.columns:
        pivot_acc = df_c.pivot_table(
            index="corruption",
            columns=aug_col,
            values="eval/acc",
            aggfunc="mean",
        )
        # 按统一顺序重排列
        cols = [a for a in AUG_ORDER if a in pivot_acc.columns]
        pivot_acc = pivot_acc[cols]

        plt.figure(figsize=(10, 8))
        im = plt.imshow(pivot_acc.values, aspect="auto")
        plt.colorbar(im, label="Accuracy")
        plt.xticks(
            ticks=np.arange(len(pivot_acc.columns)),
            labels=pivot_acc.columns,
            rotation=45,
            ha="right",
        )
        plt.yticks(
            ticks=np.arange(len(pivot_acc.index)),
            labels=pivot_acc.index,
        )
        plt.title(f"{dataset}: CIFAR-C Accuracy Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_heatmap_cifar_c_acc.png"))
        plt.close()

    if "eval/ece" in df_c.columns:
        pivot_ece = df_c.pivot_table(
            index="corruption",
            columns=aug_col,
            values="eval/ece",
            aggfunc="mean",
        )
        cols = [a for a in AUG_ORDER if a in pivot_ece.columns]
        pivot_ece = pivot_ece[cols]

        plt.figure(figsize=(10, 8))
        im = plt.imshow(pivot_ece.values, aspect="auto")
        plt.colorbar(im, label="ECE")
        plt.xticks(
            ticks=np.arange(len(pivot_ece.columns)),
            labels=pivot_ece.columns,
            rotation=45,
            ha="right",
        )
        plt.yticks(
            ticks=np.arange(len(pivot_ece.index)),
            labels=pivot_ece.index,
        )
        plt.title(f"{dataset}: CIFAR-C ECE Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_heatmap_cifar_c_ece.png"))
        plt.close()

    # ==========================
    # E) Combined summary (clean + CIFAR-C)
    # ==========================
    
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

    combined = clean_summary.add_prefix("clean_").join(
        summary_c.add_prefix("cifar_c_"), how="outer"
    )

    # 重排顺序
    combined = combined.reindex(AUG_ORDER, copy=False)
    combined.index = [AUG_NAME_MAP.get(a, a) for a in combined.index]

    table = combined.reset_index().rename(
        columns={
            "index": "Augmentation",
            "clean_eval/acc": "Clean Acc.",
            "clean_eval/ece": "Clean ECE",
            "clean_eval/mce": "Clean MCE",
            "clean_eval/loss": "Clean Loss",
            "cifar_c_mean_acc": "mCA",
            "cifar_c_mean_ece": "Mean ECE",
            "cifar_c_mean_mce": "Mean MCE",
            "cifar_c_mean_loss": "Mean Loss",
        }
    )

    csv_out = os.path.join(out_dir, f"stage3_summary_{dataset}.csv")
    table.to_csv(csv_out, index=False)
    print(f"\n[INFO] Combined summary CSV saved to: {csv_out}")

    tex_out = os.path.join(out_dir, f"stage3_summary_{dataset}.tex")
    table.to_latex(
        tex_out,
        index=False,
        float_format="%.4f",
        escape=True,
        caption=f"Stage-3 clean + CIFAR-C summary on {dataset.upper()}.",
        label=f"tab:stage3_{dataset}",
    )
    print(f"[INFO] LaTeX table saved to: {tex_out}")

    # ==========================
    # F) metric × corruption 表 & 柱状图（和之前一样）
    # ==========================
    print("\n=== Stage-3: metric × corruption (dataset={}) ===".format(dataset))

    metrics_for_mc = [m for m in ["eval/acc", "eval/ece", "eval/mce", "eval/loss"]
                      if m in df.columns]

    clean_means = {m: df_clean[m].mean() for m in metrics_for_mc}
    corruption_order = sorted(df_c["corruption"].unique())
    mc_rows = {}
    for m in metrics_for_mc:
        series_c = df_c.groupby("corruption")[m].mean()
        row_vals = [clean_means[m]] + [series_c.get(c, np.nan) for c in corruption_order]
        mc_rows[m] = row_vals

    columns = ["clean"] + corruption_order
    metric_corruption_table = pd.DataFrame.from_dict(
        mc_rows, orient="index", columns=columns
    )

    print(metric_corruption_table)

    # ---------- 保存 CSV ----------
    mc_csv_out = os.path.join(out_dir, f"stage3_metrics_by_corruption_{dataset}.csv")
    metric_corruption_table.to_csv(mc_csv_out)
    print(f"[INFO] Metric×corruption CSV saved to: {mc_csv_out}")

    # ---------- 保存格式更友好的 LaTeX 表 ----------
    pretty_table = metric_corruption_table.rename(
        index={
            "eval/acc": "Accuracy",
            "eval/ece": "ECE",
            "eval/mce": "MCE",
            "eval/loss": "Loss",
        }
    )
    pretty_table.index.name = "Metric"  # 让 index 这一列叫 Metric

    mc_tex_out = os.path.join(out_dir, f"stage3_metrics_by_corruption_{dataset}.tex")
    pretty_table.to_latex(
        mc_tex_out,
        float_format="%.4f",
        escape=True,
        caption=f"{dataset.upper()} metrics by corruption (clean + CIFAR-C).",
        label=f"tab:stage3_corr_{dataset}",
    )
    print(f"[INFO] Metric×corruption LaTeX saved to: {mc_tex_out}")

    for m in metrics_for_mc:
        vals = metric_corruption_table.loc[m].values
        plt.figure(figsize=(12, 4))
        plt.bar(columns, vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(m)
        plt.title(f"{dataset}: {m} by corruption (clean + CIFAR-C)")
        plt.tight_layout()
        short_name = m.replace("eval/", "").replace("/", "_")
        fig_path = os.path.join(out_dir, f"{dataset}_bycorr_{short_name}.png")
        plt.savefig(fig_path)
        plt.close()

    print(f"[INFO] All plots & tables saved under: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="analysis/stage3_all_runs.csv",
        help="Path to CSV exported by export_wandb_runs.py",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        required=True,
        help="Which dataset to analyze.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis/plots_stage3",
        help="Directory to save plots and summary tables.",
    )
    args = parser.parse_args()
    analyze(args.csv, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()