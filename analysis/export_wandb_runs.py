# export_wandb_runs.py
# EN: Export selected metrics from a W&B project to a CSV file (Stage-1/2/3).
# ZH: 从 W&B 项目中导出指定指标到 CSV，用于离线分析（Stage-1/2/3 通用）。

import os
import argparse
from typing import Any, Dict, Optional, List

import wandb
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------
# Fixed corruption subset for MedMNIST-C export (thesis contract)
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
# Helpers: robust config access (flattened or nested)
# ---------------------------------------------------------
def _cfg_get(cfg: Dict[str, Any], key: str, default=None):
    """
    EN: Get config value supporting both flattened keys ("a.b") and nested dict.
    ZH: 同时支持扁平 key（"a.b"）与嵌套 dict 的取值方式。
    """
    if key in cfg:
        return cfg.get(key, default)

    # nested lookup for dot keys
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ---------------------------------------------------------
# Helper: infer augmentation label (mainly for Stage-2 training)
# ---------------------------------------------------------
def infer_aug_label(cfg: Dict[str, Any], raw_aug: Optional[str]) -> Optional[str]:
    """
    EN:
      For Stage-2 runs, data.aug can be "baseline" but real augmentation
      is controlled via mixup/cutmix alpha. Map baseline + alpha>0 => mixup/cutmix.
    ZH:
      Stage-2 中 data.aug 可能仍为 baseline，但 mixup/cutmix 由 alpha 打开。
      baseline + alpha>0 => mixup/cutmix。
    """
    if raw_aug is None:
        return None

    if str(raw_aug).lower() != "baseline":
        return raw_aug

    # support both flattened and nested
    mixup_alpha = _safe_float(_cfg_get(cfg, "train.mixup_alpha", _cfg_get(cfg, "train", {}).get("mixup_alpha", 0.0)))
    cutmix_alpha = _safe_float(_cfg_get(cfg, "train.cutmix_alpha", _cfg_get(cfg, "train", {}).get("cutmix_alpha", 0.0)))

    if mixup_alpha > 0 and cutmix_alpha > 0:
        # in your pipeline you prefer labeling as "mixup" if both appear
        return "mixup"
    if mixup_alpha > 0:
        return "mixup"
    if cutmix_alpha > 0:
        return "cutmix"
    return raw_aug


# ---------------------------------------------------------
# Detect stage / dataset / metadata in a robust way
# ---------------------------------------------------------
def infer_dataset(cfg: Dict[str, Any]) -> Optional[str]:
    dataset = (
        _cfg_get(cfg, "dataset")  # often Stage-3
        or _cfg_get(cfg, "data.name")
        or _cfg_get(cfg, "data", {}).get("name") if isinstance(_cfg_get(cfg, "data"), dict) else None
    )
    return str(dataset).lower() if dataset is not None else None


def infer_stage(cfg: Dict[str, Any]) -> Optional[str]:
    stage = (
        _cfg_get(cfg, "stage")
        or _cfg_get(cfg, "train.stage")
        or _cfg_get(cfg, "meta.stage")
        or "unknown"
    )
    return str(stage).lower() if stage is not None else None


def infer_exp_id(cfg: Dict[str, Any]) -> Optional[str]:
    exp_id = _cfg_get(cfg, "train.exp_id", None)
    if exp_id is None:
        exp_id = _cfg_get(cfg, "exp_id", None)
    return str(exp_id) if exp_id is not None else None


def infer_lr(cfg: Dict[str, Any]) -> Optional[float]:
    lr = _cfg_get(cfg, "train.lr", None)
    if lr is None:
        lr = _cfg_get(cfg, "lr", None)
    return _safe_float(lr, default=None) if lr is not None else None


def infer_aug(cfg: Dict[str, Any]) -> Optional[str]:
    raw_aug = (
        _cfg_get(cfg, "augmentation")              # often Stage-3
        or _cfg_get(cfg, "data.aug")
        or (_cfg_get(cfg, "data", {}).get("aug") if isinstance(_cfg_get(cfg, "data"), dict) else None)
    )
    aug = infer_aug_label(cfg, raw_aug)
    return str(aug).lower() if aug is not None else None


def infer_corruption(summary: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[str]:
    # prefer summary meta (your stage3 logger), fallback to config meta
    corr = summary.get("meta/corruption")
    if corr is None:
        corr = _cfg_get(cfg, "meta.corruption", None)
    return str(corr).lower() if corr is not None else None


def infer_severity(summary: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    sev = summary.get("meta/severity")
    if sev is None:
        sev = _cfg_get(cfg, "meta.severity", 0)
    return _safe_int(sev, default=0)


# ---------------------------------------------------------
# Which metrics to export
# ---------------------------------------------------------
STAGE2_3_METRICS = ["eval/acc", "eval/bal_acc", "eval/ece", "eval/mce", "eval/loss"]

# Stage-1 cares mainly about validation curves/summaries
# We'll export all keys that start with "val/" plus a few helpful ones if present
def collect_stage1_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in summary.items():
        if isinstance(k, str) and k.startswith("val/"):
            out[k] = v
    # also export eval/* if present (sometimes you logged eval on val split)
    for k in STAGE2_3_METRICS:
        if k in summary and k not in out:
            out[k] = summary.get(k)
    return out


# ---------------------------------------------------------
# Core export
# ---------------------------------------------------------
def export_project_runs(
    project: str,
    entity: str,
    outfile: str,
    mode: str = "auto",                 # auto | stage1 | stage2 | stage3
    group_prefix: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    stage_filter: Optional[str] = None,
    exp_ids: Optional[List[str]] = None,
    keep_medmnistc8: bool = True,
):
    """
    EN:
      Export runs from W&B project with flexible filtering for Stage-1/2/3.
    ZH:
      从 W&B 项目导出 runs，支持 Stage-1/2/3 的筛选与统一列结构。
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    rows = []

    for run in tqdm(runs, desc=f"Exporting runs from {entity}/{project}"):
        grp = (run.group or "") if hasattr(run, "group") else ""
        if group_prefix is not None and not grp.startswith(group_prefix):
            continue

        cfg = dict(run.config or {})
        summary = dict(run.summary or {})

        dataset = infer_dataset(cfg)
        stage = infer_stage(cfg)
        exp_id = infer_exp_id(cfg)
        lr = infer_lr(cfg)
        aug = infer_aug(cfg)

        # optional user filters
        if dataset_filter is not None and dataset != dataset_filter.lower():
            continue
        if stage_filter is not None and stage != stage_filter.lower():
            continue
        if exp_ids is not None and exp_id not in exp_ids:
            continue

        # auto mode: infer by presence of corruption meta or exp_id grid
        corr = infer_corruption(summary, cfg)
        sev = infer_severity(summary, cfg)

        if mode == "auto":
            if exp_id in ["C1", "C2", "C3", "C4"] and (corr is None or corr == "clean" or sev == 0):
                inferred_mode = "stage1"
            elif corr is not None:
                # likely stage3 evaluation
                inferred_mode = "stage3"
            else:
                inferred_mode = "stage2"
        else:
            inferred_mode = mode.lower()

        # Stage-1: keep only exp_id C1-4 by default
        if inferred_mode == "stage1":
            if exp_id not in ["C1", "C2", "C3", "C4"]:
                continue

        # Stage-3: optionally filter MedMNIST-C to fixed 8 corruptions (keep clean always)
        is_clean = (corr == "clean") or (sev == 0)
        if keep_medmnistc8 and inferred_mode == "stage3":
            if dataset in ["dermamnist", "pathmnist"] and (not is_clean):
                if corr not in MEDMNISTC_8:
                    continue

        # only finished runs (consistent with your analyses)
        state = getattr(run, "state", None)
        if state is not None and state != "finished":
            continue

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "group": grp,
            "dataset": dataset,
            "stage": stage,
            "mode": inferred_mode,
            "augmentation": aug,
            "exp_id": exp_id,
            "lr": lr,
            "seed": _cfg_get(cfg, "seed", _cfg_get(cfg, "train.seed", None)),
            # corruption meta
            "meta/corruption": corr,
            "meta/severity": sev,
            # run state
            "state": state,
        }

        # metrics
        if inferred_mode == "stage1":
            row.update(collect_stage1_metrics(summary))
        else:
            for k in STAGE2_3_METRICS:
                row[k] = summary.get(k)

        rows.append(row)

    if not rows:
        print("[WARN] No runs exported. Check project / filters.")
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
        description="Export W&B project runs to CSV (Stage-1/2/3)."
    )
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY", ""), help="W&B entity")
    parser.add_argument("--outfile", required=True, help="Output CSV path")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "stage1", "stage2", "stage3"],
        help="Export mode (default: auto)",
    )
    parser.add_argument("--group-prefix", default=None, help="Only export runs with group prefix")
    parser.add_argument("--dataset", default=None, help="Filter by dataset (cifar10/cifar100/dermamnist/pathmnist)")
    parser.add_argument("--stage", default=None, help="Filter by stage (stage1/stage2/stage3 if logged)")
    parser.add_argument("--exp-ids", default=None, help="Comma-separated exp_id list, e.g. C1,C2,C3,C4")
    parser.add_argument(
        "--no-medmnistc8",
        action="store_true",
        help="Disable MedMNIST-C fixed 8-corruption filter (keep all corruptions).",
    )

    args = parser.parse_args()
    if not args.entity:
        raise SystemExit("Please set --entity or export WANDB_ENTITY in your shell.")

    exp_ids = None
    if args.exp_ids:
        exp_ids = [s.strip() for s in args.exp_ids.split(",") if s.strip()]

    export_project_runs(
        project=args.project,
        entity=args.entity,
        outfile=args.outfile,
        mode=args.mode,
        group_prefix=args.group_prefix,
        dataset_filter=args.dataset,
        stage_filter=args.stage,
        exp_ids=exp_ids,
        keep_medmnistc8=(not args.no_medmnistc8),
    )


if __name__ == "__main__":
    main()