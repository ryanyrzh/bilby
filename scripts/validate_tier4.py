#!/usr/bin/env python
"""Validate Tier 4 smoke / pilot outputs.

Usage:
  python scripts/validate_tier4.py OUTDIR [CHECKS...]

CHECKS may be legacy aliases (agn, generic, cmp) or explicit:
  pe:LABEL   -> require OUTDIR/LABEL_result.{json,pickle}
  cmp:LABEL  -> require OUTDIR/LABEL_summary.json
"""
import json
import math
import os
import sys


def finite(x):
    return x is not None and math.isfinite(float(x))


def check_pe_result(outdir, label):
    patterns = [
        os.path.join(outdir, f"{label}_result.json"),
        os.path.join(outdir, f"{label}_result.pickle"),
    ]
    if not any(os.path.exists(p) for p in patterns):
        raise FileNotFoundError(f"Missing result for {label} in {outdir}")
    json_path = os.path.join(outdir, f"{label}_result.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        for key in ("log_evidence", "log_bayes_factor"):
            if key in data and not finite(data[key]):
                raise ValueError(f"{label}: {key} not finite ({data.get(key)})")
    print(f"PASS: {label}")


def check_comparison(outdir, label):
    path = os.path.join(outdir, f"{label}_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        summary = json.load(f)
    for model in ("agn", "generic"):
        if model not in summary:
            raise KeyError(f"Missing model {model} in {path}")
        vals = summary[model]
        for key in ("log_bf_lensed_vs_simple", "log10_bf_lensed_vs_simple"):
            if key not in vals or not finite(vals[key]):
                raise ValueError(f"{model}: {key} not finite ({vals.get(key)})")
    print(f"PASS: {label}_summary.json")


def resolve_check(name):
    aliases = {
        "agn": ("pe", "agn_smoke"),
        "generic": ("pe", "generic_smoke"),
        "cmp": ("cmp", "cmp_smoke"),
    }
    if name in aliases:
        return aliases[name]
    if name.startswith("pe:") or name.startswith("cmp:"):
        kind, label = name.split(":", 1)
        if not label:
            raise ValueError(f"Empty label in check: {name}")
        return kind, label
    raise ValueError(
        f"Unknown check: {name}. Use agn|generic|cmp or pe:LABEL|cmp:LABEL"
    )


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "outdir_test"
    checks = sys.argv[2:] or ["agn", "generic", "cmp"]
    for name in checks:
        kind, label = resolve_check(name)
        if kind == "pe":
            check_pe_result(outdir, label)
        elif kind == "cmp":
            check_comparison(outdir, label)
        else:
            raise ValueError(f"Unknown check kind: {kind}")
    print(f"Tier 4 validation ({', '.join(checks)}): PASS")


if __name__ == "__main__":
    main()
