# export_wandb_runs.py
# EN: Export selected metrics from a W&B project to a CSV file.
# ZH: 从 W&B 项目中导出指定指标到 CSV，用于后续离线分析（Stage-2 / Stage-3 通用）。

import os
import argparse
import wandb
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------
# Fixed corruption subset for MedMNIST-C export (your thesis contract)
# ---------------------------------------------------------
MEDMNISTC_8 = [
    "defocus_blur",
    "motion_blur",
    "pixelate",
    "jpeg_compression",
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "contrast_down",
]


# ---------------------------------------------------------
# Helper: infer augmentation label (for Stage-2 training)
# (KEEP if you still export Stage-2 and want mixup/cutmix correctly labeled)
# ---------------------------------------------------------
def infer_aug_label(cfg: dict, raw_aug: str | None) -> str | None:
    """
    EN:
        For Stage-2 runs, data.aug is often "baseline" and the real augmentation
        is controlled via train.mixup_alpha / train.cutmix_alpha.
        This helper maps:
            baseline + mixup_alpha>0  -> "mixup"
            baseline + cutmix_alpha>0 -> "cutmix"
        Otherwise returns raw_aug.

    ZH:
        Stage-2 中，很多 YAML 里 data.aug 仍写的是 "baseline"，
        真正的增强是通过 train.mixup_alpha / train.cutmix_alpha 打开的。
        这个函数把：
            baseline + mixup_alpha>0  映射成 "mixup"
            baseline + cutmix_alpha>0 映射成 "cutmix"
        其他情况直接返回原始 raw_aug。
    """
    if raw_aug is None:
        return None

    if str(raw_aug) != "baseline":
        return raw_aug

    # NOTE:
    # W&B config is often flattened; but your current pipeline might log nested dict.
    # We keep your original logic to avoid over-touching.
    train_cfg = cfg.get("train", {})
    mixup_alpha = train_cfg.get("mixup_alpha", 0.0)
    cutmix_alpha = train_cfg.get("cutmix_alpha", 0.0)

    try:
        mixup_alpha = float(mixup_alpha)
    except Exception:
        mixup_alpha = 0.0
    try:
        cutmix_alpha = float(cutmix_alpha)
    except Exception:
        cutmix_alpha = 0.0

    if mixup_alpha > 0 and cutmix_alpha > 0:
        return "mixup"
    if mixup_alpha > 0:
        return "mixup"
    if cutmix_alpha > 0:
        return "cutmix"

    return raw_aug


# ---------------------------------------------------------
# Core export function
# ---------------------------------------------------------
def export_project_runs(project: str,
                        entity: str,
                        outfile: str,
                        group_prefix: str | None = None):
    """
    EN:
        Export all runs from `entity/project` whose group starts with
        `group_prefix` (if given), and save them as `outfile` (CSV).

    ZH:
        从 `entity/project` 中导出所有 run（可选：只导出 group
        以 group_prefix 开头的），保存为 `outfile` CSV。
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    rows = []

    for run in tqdm(runs, desc=f"Exporting runs from {project}"):
        grp = (run.group or "") if hasattr(run, "group") else ""
        if group_prefix is not None and not grp.startswith(group_prefix):
            continue

        cfg = dict(run.config)
        summary = dict(run.summary)

        # -----------------------------
        # dataset: Stage-3 vs Stage-2
        # -----------------------------
        dataset = (
            cfg.get("dataset")  # Stage-3
            or cfg.get("data.name")
            or cfg.get("data", {}).get("name")
        )
        dataset_str = str(dataset).lower() if dataset is not None else None

        # -----------------------------
        # augmentation
        # -----------------------------
        raw_aug = (
            cfg.get("augmentation")               # Stage-3
            or cfg.get("data.aug")               # some configs
            or cfg.get("data", {}).get("aug")    # Stage-2 / Stage-1
        )
        aug = infer_aug_label(cfg, raw_aug)
        aug_str = str(aug).lower() if aug is not None else None

        # stage tag (optional)
        stage = (
            cfg.get("stage")
            or cfg.get("train.stage", "unknown")
        )

        # -----------------------------
        # ONLY CHANGE YOU REQUESTED:
        # If this is MedMNIST Stage-3 data, export only the fixed 8 corruptions.
        # We rely on meta/corruption stored in SUMMARY (as in your Stage-3 code).
        # -----------------------------
        corr = summary.get("meta/corruption")
        sev = summary.get("meta/severity")

        corr_str = str(corr).lower() if corr is not None else None

        is_medmnist = dataset_str in ["dermamnist", "pathmnist"]
        is_stage3 = str(stage).lower() == "stage3"

        # Keep clean always (corr == "clean" or severity == 0)
        is_clean = (corr_str == "clean") or (sev == 0)

        if is_medmnist and is_stage3 and (not is_clean):
            # Filter to the shared fixed 8 corruptions
            if corr_str not in MEDMNISTC_8:
                continue

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "group": grp,
            "dataset": dataset_str,
            "augmentation": aug_str,
            "stage": str(stage).lower() if stage is not None else None,
            # Main metrics
            "eval/acc": summary.get("eval/acc"),
            "eval/ece": summary.get("eval/ece"),
            "eval/mce": summary.get("eval/mce"),
            "eval/loss": summary.get("eval/loss"),
            "eval/bal_acc": summary.get("eval/bal_acc"),
            # For Stage-3 (CIFAR-C / MedMNIST-C)
            "meta/corruption": corr,
            "meta/severity": sev,
            # run state
            "state": getattr(run, "state", None),
        }

        rows.append(row)

    if not rows:
        print("[WARN] No runs exported. Check project / group_prefix.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_csv(outfile, index=False)
    print(f"[DONE] Exported {len(df)} rows to: {outfile}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export W&B project runs to CSV (Stage-2 / Stage-3)."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project name, e.g. stage2-aug-comparison / stage3-robustness",
    )
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY", ""),
        help="W&B entity (default: from WANDB_ENTITY env)",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        help="Output CSV path, e.g. analysis/stage3_all_runs.csv",
    )
    parser.add_argument(
        "--group-prefix",
        default=None,
        help="Only export runs whose group starts with this prefix",
    )

    args = parser.parse_args()

    if not args.entity:
        raise SystemExit("Please set --entity or export WANDB_ENTITY in your shell.")

    export_project_runs(
        project=args.project,
        entity=args.entity,
        outfile=args.outfile,
        group_prefix=args.group_prefix,
    )


if __name__ == "__main__":
    main()