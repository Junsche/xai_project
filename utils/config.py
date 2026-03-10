# utils/config.py
import argparse
import yaml
import copy


def _smart_cast(v: str):
    """
    Try to cast a string using YAML semantics.
    This allows numbers, booleans, lists, etc. to be parsed correctly.
    Falls back to the raw string if parsing fails.
    """
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def parse_with_overrides():
    """
    Parse YAML configuration files and apply command-line overrides.

    Expected usage:
        python main.py base.yaml dataset.yaml aug.yaml \
            --override train.lr=0.05 model.name=resnet18
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="+",
        help="YAML files in order: base, dataset, augmentation"
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Key-value pairs to override config values (e.g., train.lr=0.05)"
    )
    args = parser.parse_args()

    # Load and merge YAML files sequentially
    cfg = {}
    for y in args.configs:
        with open(y, "r") as f:
            part = yaml.safe_load(f)
        cfg = _deep_merge(cfg, part)

    # Apply command-line overrides
    for kv in args.override:
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        value = _smart_cast(value)
        _assign(cfg, key.split("."), value)

    return cfg


def _deep_merge(a, b):
    """
    Recursively merge dictionary b into dictionary a.
    Values in b take precedence over values in a.
    """
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _assign(d, keys, value):
    """
    Assign a value into a nested dictionary given a list of keys.

    Example:
        keys = ["train", "lr"]
        value = 0.05
        -> d["train"]["lr"] = 0.05
    """
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value