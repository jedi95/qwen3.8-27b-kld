#!/usr/bin/env python3
"""Verify a fresh campaign run against committed expected comparison numbers.

Usage: verify_repro.py <expected-dir> <actual-dir> [--kld-tol T] [--top1-tol T]

Compares mean_kld / top1_agreement of every expected comparison JSON against
the rerun's JSON of the same name. Exit 0 iff all rows pass:
  - floor_probe* rows must be <= 1e-6 mean KLD (acceptance gate, see
    docs/METHODOLOGY.md §6: any nonzero floor invalidates the ladder);
  - candidate rows pass within tolerances (defaults: |Δ mean_kld| <= 0.002,
    |Δ top1| <= 0.005). Tighten to 0 for a same-GPU bit-identity claim.
Rows the rerun could not produce (load errors) are reported as MISSING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expected")
    ap.add_argument("actual")
    ap.add_argument("--kld-tol", type=float, default=0.002)
    ap.add_argument("--top1-tol", type=float, default=0.005)
    args = ap.parse_args()

    exp_dir, act_dir = Path(args.expected), Path(args.actual)
    fails = 0
    rows = sorted(exp_dir.glob("*.json"))
    print(f"{'row':44s} {'mean_kld exp→act':>28s} {'top1 exp→act':>20s}  verdict")
    for ep in rows:
        name = ep.stem
        ap_ = act_dir / ep.name
        exp = json.loads(ep.read_text())
        if "error" in exp:
            print(f"{name:44s} {'(expected error row)':>28s} {'':>20s}  SKIP")
            continue
        if not ap_.exists():
            print(f"{name:44s} {'':>28s} {'':>20s}  MISSING")
            fails += 1
            continue
        act = json.loads(ap_.read_text())
        if "error" in act:
            print(f"{name:44s} {'':>28s} {'':>20s}  ERROR: {act['error']}")
            fails += 1
            continue
        dk = abs(exp["mean_kld"] - act["mean_kld"])
        dt = abs(exp["top1_agreement"] - act["top1_agreement"])
        if name.startswith("floor"):
            ok = act["mean_kld"] <= 1e-6
            why = "" if ok else " (floor gate)"
        else:
            ok = dk <= args.kld_tol and dt <= args.top1_tol
            why = ""
        verdict = "PASS" if ok else f"FAIL{why}"
        fails += 0 if ok else 1
        print(f"{name:44s} {exp['mean_kld']:>12.6f} → {act['mean_kld']:<14.6f} "
              f"{exp['top1_agreement']:>8.4f} → {act['top1_agreement']:<8.4f}  {verdict}")
    print(f"\n{len(rows) - fails}/{len(rows)} rows pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
