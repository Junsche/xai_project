# export_wandb_runs.py
# EN: Export selected metrics from a W&B project to a CSV file.
# ZH: 从 W&B 项目中导出指定指标到 CSV，用于后续离线分析（Stage-2 / Stage-3 通用）。

import os
import argparse
import wandb
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------
# Helper: infer augmentation label (for Stage-2 training)
# ---------------------------------------------------------
def infer_aug_label(cfg: dict, raw_aug: str | None) -> str | None:
    """
    EN:
        For Stage-2 runs, data.aug is often "baseline" and the real
        augmentation is controlled via train.mixup_alpha / train.cutmix_alpha.
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

    # 只在 baseline 情况下做“重命名”
    if str(raw_aug) != "baseline":
        return raw_aug

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
        # 如果你确实有二者同时开的情况，可以改成 "mixup+cutmix"
        return "mixup"
    if mixup_alpha > 0:
        return "mixup"
    if cutmix_alpha > 0:
        return "cutmix"

    # 纯 baseline
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
        # Optional filter by group prefix
        grp = (run.group or "") if hasattr(run, "group") else ""
        if group_prefix is not None and not grp.startswith(group_prefix):
            continue

        cfg = dict(run.config)
        summary = dict(run.summary)

        # -----------------------------
        # dataset: Stage-3 vs Stage-2
        # -----------------------------
        # Stage-3 (run_stage3_eval.py):
        #   config: {"dataset": "...", "augmentation": "...", "stage": "stage3", ...}
        #
        # Stage-2 / Stage-1 (main.py):
        #   config: {"data": {"name": "...", "aug": "baseline", ...}, ...}
        dataset = (
            cfg.get("dataset")  # Stage-3
            or cfg.get("data.name")
            or cfg.get("data", {}).get("name")
        )

        # -----------------------------
        # augmentation: with inference for mixup/cutmix
        # -----------------------------
        raw_aug = (
            cfg.get("augmentation")               # Stage-3
            or cfg.get("data.aug")               # some configs
            or cfg.get("data", {}).get("aug")    # Stage-2 / Stage-1
        )
        aug = infer_aug_label(cfg, raw_aug)

        # stage tag (optional)
        stage = (
            cfg.get("stage")
            or cfg.get("train.stage", "unknown")
        )

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "group": grp,
            "dataset": dataset,
            "augmentation": aug,
            "stage": stage,
            # Main metrics
            "eval/acc": summary.get("eval/acc"),
            "eval/ece": summary.get("eval/ece"),
            "eval/mce": summary.get("eval/mce"),
            "eval/loss": summary.get("eval/loss"),
            "eval/bal_acc": summary.get("eval/bal_acc"),
            # For Stage-3 (CIFAR-C)
            "meta/corruption": summary.get("meta/corruption"),
            "meta/severity": summary.get("meta/severity"),
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
        help="W&B project name, e.g. robustness-stage3 or aug-comparison",
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