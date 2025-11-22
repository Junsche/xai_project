# utils/config.py（只示意与 override 相关的部分，无需改动你现有逻辑的其他段）
import argparse, yaml, copy

def _smart_cast(v: str):
    """EN: Try YAML to cast numbers/bools; fallback to raw string.
       ZH: 用 YAML 尝试把字符串转成数值/布尔；失败则原样返回。"""
    try:
        return yaml.safe_load(v)
    except Exception:
        return v

def parse_with_overrides():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="YAML files in order: base, dataset, aug")
    parser.add_argument("--override", nargs="*", default=[], help="k=v pairs to override (e.g., train.lr=0.05)")
    args = parser.parse_args()

    # load and merge yamls / 依次合并 YAML
    cfg = {}
    for y in args.configs:
        with open(y, "r") as f:
            part = yaml.safe_load(f)
        cfg = _deep_merge(cfg, part)

    # apply overrides / 应用覆盖项
    for kv in args.override:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        v = _smart_cast(v)
        _assign(cfg, k.split("."), v)

    return cfg

def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _assign(d, keys, value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value