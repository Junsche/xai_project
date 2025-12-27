# -*- coding: utf-8 -*-
"""
Analyze Stage-3 results (clean + corruption benchmarks) exported from W&B into a CSV.

Supports:
- CIFAR10/100: clean test + CIFAR-C
- DermaMNIST/PathMNIST: clean test + MedMNIST-C

Paper-style aggregation for corruption metrics:
1) for each (aug, corruption): mean over severities (1..5)
2) for each aug: mean over corruptions
=> mCA / mECE / mMCE / mLoss (and optionally mBalAcc)

Outputs (per dataset):
- stage3_clean_summary_{dataset}.csv/.tex
- stage3_bycorr_sevmean_{dataset}.csv/.tex              (per aug, per corruption)
- stage3_overall_corrmean_{dataset}.csv/.tex            (per aug overall: mCA etc.)
- stage3_paper_main_table_{dataset}.csv/.tex            (clean + mCA in one table)
- figures (PDF + PNG): clean acc, mCA, heatmap, severity curve (optional)

Note:
- This script assumes your exported CSV contains columns:
  dataset, augmentation, eval/acc, eval/ece, eval/mce, eval/loss, eval/bal_acc (optional),
  meta/corruption, meta/severity, state (optional).
"""

import os
import argparse
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Global style (paper-ish)
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})


# -----------------------------
# Augmentation ordering / naming
# -----------------------------
AUG_ORDER = [
    "baseline",
    "rotation_erasing",
    "autoaugment",
    "randaugment",
    "mixup",
    "cutmix",
    "styleaug",
    "diffusemix",
    "augmix",
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
    "augmix": "AugMix",
}


# -----------------------------
# Helpers
# -----------------------------
def _ensure_corruption_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has columns: corruption, severity (copied from meta/* if needed)."""
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

    # normalize types
    df["corruption"] = df["corruption"].astype(str)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0).astype(int)
    return df


def _metric_cols(df: pd.DataFrame) -> List[str]:
    return [m for m in ["eval/acc", "eval/bal_acc", "eval/ece", "eval/mce", "eval/loss"] if m in df.columns]


def _ordered_augs_present(index_like) -> List[str]:
    present = set(index_like)
    return [a for a in AUG_ORDER if a in present]


def _paper_style_corruption_summary(df_c: pd.DataFrame, aug_col: str):
    """
    Paper-style aggregation:
      bycorr: per (aug, corruption): mean over severities
      overall: per aug: mean over corruptions of those severity-means
    """
    metrics = _metric_cols(df_c)
    if not metrics:
        raise ValueError("No metric columns found (expected eval/acc etc.).")

    # 1) per (aug, corruption): mean over severities
    bycorr = (
        df_c
        .groupby([aug_col, "corruption"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    bycorr = bycorr.rename(columns={m: f"sevmean_{m}" for m in metrics})

    # 2) per aug: mean over corruptions
    overall = (
        bycorr
        .groupby(aug_col, dropna=False)[[f"sevmean_{m}" for m in metrics]]
        .mean()
        .reset_index()
    )

    # rename to paper-friendly names
    rename_map = {
        "sevmean_eval/acc": "mCA",
        "sevmean_eval/ece": "mECE",
        "sevmean_eval/mce": "mMCE",
        "sevmean_eval/loss": "mLoss",
        "sevmean_eval/bal_acc": "mBalAcc",
    }
    overall = overall.rename(columns=rename_map)

    return bycorr, overall


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply numeric rounding suitable for paper:
    - Acc/BalAcc: 4 decimals
    - ECE/MCE: 4 decimals
    - Loss: 4 decimals
    """
    out = df.copy()
    for c in out.columns:
        if c in ["Augmentation", "augmentation", "corruption"]:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float)

    # round common columns
    for c in ["Clean Acc.", "Clean BalAcc", "mCA", "mBalAcc"]:
        if c in out.columns:
            out[c] = out[c].round(4)
    for c in ["Clean ECE", "Clean MCE", "mECE", "mMCE"]:
        if c in out.columns:
            out[c] = out[c].round(4)
    for c in ["Clean Loss", "mLoss"]:
        if c in out.columns:
            out[c] = out[c].round(4)

    # also round any remaining numeric cols
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]) and c not in ["severity"]:
            out[c] = out[c].round(4)

    return out


def _save_csv_tex(df: pd.DataFrame, csv_path: str, tex_path: str, caption: str, label: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # LaTeX
    df.to_latex(
        tex_path,
        index=False,
        float_format="%.4f",
        escape=True,
        caption=caption,
        label=label,
    )


def _barplot(values: pd.Series, title: str, ylabel: str, out_prefix: str):
    """Save bar plot as PDF + PNG."""
    plt.figure(figsize=(9, 4.6))
    plt.bar(values.index, values.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_prefix + ".pdf")
    plt.savefig(out_prefix + ".png", dpi=200)
    plt.close()


def _heatmap(pivot: pd.DataFrame, title: str, out_prefix: str, cbar_label: str):
    plt.figure(figsize=(10.5, 8.5))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label=cbar_label)
    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=pivot.columns,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=pivot.index,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_prefix + ".pdf")
    plt.savefig(out_prefix + ".png", dpi=200)
    plt.close()


def analyze(csv_path: str, dataset: str, out_dir: str = "analysis/plots_stage3"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns:
        raise ValueError("CSV missing required column: dataset")

    df = df[df["dataset"].astype(str).str.lower() == dataset.lower()].copy()
    if df.empty:
        print(f"[WARN] No rows for dataset={dataset} in {csv_path}")
        return

    if "state" in df.columns:
        df = df[df["state"] == "finished"].copy()

    df = _ensure_corruption_severity(df)

    if "augmentation" not in df.columns:
        raise ValueError("CSV missing required column: augmentation")
    aug_col = "augmentation"

    # split clean vs corruption
    clean_mask = (df["corruption"] == "clean") | (df["severity"] == 0)
    df_clean = df[clean_mask].copy()
    df_c = df[~clean_mask].copy()

    metrics = _metric_cols(df)

    # -----------------------------
    # A) Clean summary (per aug)
    # -----------------------------
    clean_metrics = [m for m in metrics if m in df_clean.columns]
    if not df_clean.empty and clean_metrics:
        clean_summary = df_clean.groupby(aug_col)[clean_metrics].mean()
    else:
        clean_summary = pd.DataFrame(index=[], columns=clean_metrics)

    ordered_augs = _ordered_augs_present(clean_summary.index)
    clean_summary = clean_summary.reindex(ordered_augs)

    # rename columns to paper headers
    clean_table = clean_summary.copy()
    col_rename = {
        "eval/acc": "Clean Acc.",
        "eval/bal_acc": "Clean BalAcc",
        "eval/ece": "Clean ECE",
        "eval/mce": "Clean MCE",
        "eval/loss": "Clean Loss",
    }
    clean_table = clean_table.rename(columns=col_rename).reset_index().rename(columns={aug_col: "augmentation"})
    clean_table["Augmentation"] = clean_table["augmentation"].map(lambda x: AUG_NAME_MAP.get(str(x), str(x)))
    clean_table = clean_table.drop(columns=["augmentation"])

    # reorder columns
    desired_cols = ["Augmentation", "Clean Acc.", "Clean BalAcc", "Clean ECE", "Clean MCE", "Clean Loss"]
    clean_table = clean_table[[c for c in desired_cols if c in clean_table.columns]]
    clean_table = _format_table(clean_table)

    _save_csv_tex(
        clean_table,
        csv_path=os.path.join(out_dir, f"stage3_clean_summary_{dataset}.csv"),
        tex_path=os.path.join(out_dir, f"stage3_clean_summary_{dataset}.tex"),
        caption=f"Stage-3 clean test summary on {dataset}.",
        label=f"tab:stage3_clean_{dataset}",
    )

    # plots: clean acc (and clean balacc for medmnist)
    if "Clean Acc." in clean_table.columns:
        s = clean_table.set_index("Augmentation")["Clean Acc."]
        _barplot(s, title=f"{dataset}: Clean Accuracy", ylabel="Accuracy",
                 out_prefix=os.path.join(out_dir, f"{dataset}_clean_acc"))

    if "Clean BalAcc" in clean_table.columns and clean_table["Clean BalAcc"].notna().any():
        s = clean_table.set_index("Augmentation")["Clean BalAcc"]
        _barplot(s, title=f"{dataset}: Clean Balanced Accuracy", ylabel="Balanced Accuracy",
                 out_prefix=os.path.join(out_dir, f"{dataset}_clean_balacc"))

    # -----------------------------
    # B) Corruption paper-style summaries
    # -----------------------------
    if df_c.empty:
        print("[WARN] No corruption rows found; only clean summary was generated.")
        return

    bycorr, overall = _paper_style_corruption_summary(df_c, aug_col=aug_col)

    # order augs for overall
    overall = overall.set_index(aug_col).reindex(ordered_augs).reset_index()
    overall["Augmentation"] = overall[aug_col].map(lambda x: AUG_NAME_MAP.get(str(x), str(x)))
    overall = overall.drop(columns=[aug_col])

    # keep consistent columns
    overall_cols = ["Augmentation", "mCA", "mBalAcc", "mECE", "mMCE", "mLoss"]
    overall = overall[[c for c in overall_cols if c in overall.columns]]
    overall = _format_table(overall)

    _save_csv_tex(
        overall,
        csv_path=os.path.join(out_dir, f"stage3_overall_corrmean_{dataset}.csv"),
        tex_path=os.path.join(out_dir, f"stage3_overall_corrmean_{dataset}.tex"),
        caption=f"Stage-3 corruption summary (paper-style) on {dataset}.",
        label=f"tab:stage3_overall_{dataset}",
    )

    # plot: mCA
    if "mCA" in overall.columns:
        s = overall.set_index("Augmentation")["mCA"]
        _barplot(s, title=f"{dataset}: mCA (paper-style)", ylabel="mCA",
                 out_prefix=os.path.join(out_dir, f"{dataset}_mCA"))

    # -----------------------------
    # C) Per-augmentation, per-corruption table (severity-mean)
    # -----------------------------
    bycorr_out = bycorr.copy()
    bycorr_out["Augmentation"] = bycorr_out[aug_col].map(lambda x: AUG_NAME_MAP.get(str(x), str(x)))
    bycorr_out = bycorr_out.drop(columns=[aug_col])

    # rename severity-mean metrics to nicer labels
    bycorr_rename = {
        "sevmean_eval/acc": "sevmean_acc",
        "sevmean_eval/bal_acc": "sevmean_balacc",
        "sevmean_eval/ece": "sevmean_ece",
        "sevmean_eval/mce": "sevmean_mce",
        "sevmean_eval/loss": "sevmean_loss",
    }
    bycorr_out = bycorr_out.rename(columns=bycorr_rename)

    # order augs and keep table tidy
    # (store long format CSV; it's useful for paper appendix)
    bycorr_out = _format_table(bycorr_out)

    _save_csv_tex(
        bycorr_out,
        csv_path=os.path.join(out_dir, f"stage3_bycorr_sevmean_{dataset}.csv"),
        tex_path=os.path.join(out_dir, f"stage3_bycorr_sevmean_{dataset}.tex"),
        caption=f"Stage-3 per-corruption severity-mean metrics on {dataset}.",
        label=f"tab:stage3_bycorr_{dataset}",
    )

    # -----------------------------
    # D) Main paper table: clean + mCA (and optional calibration)
    # -----------------------------
    # join clean_table and overall on Augmentation
    paper = clean_table.set_index("Augmentation").join(
        overall.set_index("Augmentation"),
        how="outer",
    ).reset_index()

    # choose what goes into main table (clean acc + mCA are must-have)
    main_cols = ["Augmentation"]
    for c in ["Clean Acc.", "Clean BalAcc", "Clean ECE"]:
        if c in paper.columns:
            main_cols.append(c)
    for c in ["mCA", "mBalAcc", "mECE", "mMCE"]:
        if c in paper.columns:
            main_cols.append(c)

    paper_main = paper[main_cols].copy()
    paper_main = _format_table(paper_main)

    _save_csv_tex(
        paper_main,
        csv_path=os.path.join(out_dir, f"stage3_paper_main_table_{dataset}.csv"),
        tex_path=os.path.join(out_dir, f"stage3_paper_main_table_{dataset}.tex"),
        caption=f"Stage-3 clean + corruption summary (main table) on {dataset}.",
        label=f"tab:stage3_main_{dataset}",
    )

    # -----------------------------
    # E) Heatmap: corruption × aug (accuracy), severity-mean
    # -----------------------------
    # Use bycorr (severity-mean) for heatmap -> stable and paper-friendly
    if "sevmean_eval/acc" in bycorr.columns:
        pivot_acc = bycorr.pivot_table(
            index="corruption",
            columns=aug_col,
            values="sevmean_eval/acc",
            aggfunc="mean",
        )

        # reorder cols by AUG_ORDER
        cols = [a for a in AUG_ORDER if a in pivot_acc.columns]
        pivot_acc = pivot_acc[cols]
        # rename to pretty names for plotting
        pivot_acc.columns = [AUG_NAME_MAP.get(c, c) for c in pivot_acc.columns]

        _heatmap(
            pivot_acc,
            title=f"{dataset}: Accuracy heatmap (sev-mean over severity)",
            out_prefix=os.path.join(out_dir, f"{dataset}_heatmap_acc_sevmean"),
            cbar_label="Accuracy",
        )

    # -----------------------------
    # F) Severity curve: mean acc vs severity (optional, all corruptions+augs)
    # -----------------------------
    if "eval/acc" in df_c.columns:
        sev_mean_acc = df_c.groupby("severity")["eval/acc"].mean().sort_index()
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(sev_mean_acc.index, sev_mean_acc.values, marker="o")
        plt.xlabel("Severity")
        plt.ylabel("Mean Accuracy")
        plt.title(f"{dataset}: Mean Accuracy vs Severity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset}_severity_curve_acc.pdf"))
        plt.savefig(os.path.join(out_dir, f"{dataset}_severity_curve_acc.png"), dpi=200)
        plt.close()

    print(f"[DONE] Saved paper-style tables and figures under: {out_dir}")


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
        choices=["cifar10", "cifar100", "dermamnist", "pathmnist"],
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

    analyze(args.csv, args.dataset.lower(), args.out_dir)


if __name__ == "__main__":
    main()