#!/usr/bin/env python3
"""Preflight every registry entry: arch, quant method, weight files.

Derives its list from configs/models.yaml (the reference and all
candidates — deduplicated) and checks each entry at its PINNED revision, not
main (branch-indexed repos keep their weights on named branches only).

Usage:
  preflight_repos.py            # audit all registry entries, exit 1 on any FAIL
  preflight_repos.py <label>    # just one label
"""
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = f"{REPO}/configs/models.yaml"
UA = {"User-Agent": "kld-campaign-preflight"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
failures = []


def get(url, binary=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read() if binary else r.read().decode()


def check(label, model, revision, quantization):
    rev = revision or "main"
    print(f"== {label}: {model}@{rev[:40]}")
    ref = rev if SHA_RE.match(rev) else f"refs/heads/{rev}"
    refq = urllib.parse.quote(ref, safe="")
    # NOTE: use the tree endpoint, not models?revision=X `siblings` — siblings
    # on non-main branches under-reports files (misses real shard weights).
    try:
        tree = json.loads(get(f"https://huggingface.co/api/models/{model}"
                              f"/tree/{refq}"))
    except urllib.error.HTTPError as e:
        print(f"   FAIL repo/revision HTTP {e}")
        failures.append(label)
        return
    wt = [f["path"] for f in tree if f["path"].endswith(".safetensors")]
    print(f"   files: {len(tree)} | weights: {len(wt)} {wt[:3]}")
    real = [w for w in wt if "cal_trace" not in w]
    if not real:
        print("   FAIL no model .safetensors weight files at this revision")
        failures.append(label)
    raw = "https://huggingface.co/" + model + f"/raw/{refq}/"
    try:
        cj = json.loads(get(raw + "config.json"))
        arch = cj.get("architectures")
        qm = (cj.get("quantization_config") or {}).get("quant_method")
        print(f"   arch: {arch} | quant_method: {qm}"
              + (f" | registry says: {quantization}" if quantization else ""))
        if quantization and quantization != "null" and qm != quantization:
            print(f"   NOTE registry quantization={quantization!r} but config says "
                  f"{qm!r} (explicit override — verify the engine honours it)")
    except Exception as e:
        print(f"   FAIL config.json: {e}")
        failures.append(label)


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = yaml.safe_load(open(CFG))
    entries = [(cfg["reference"]["label"], cfg["reference"]["model"],
                cfg["reference"]["revision"], None)]
    seen = set()
    for c in cfg["candidates"]:
        key = (c["model"], c.get("revision"))
        if key in seen and not only:
            continue  # cutlass twins share weights; one check per revision
        seen.add(key)
        entries.append((c["label"], c["model"], c.get("revision"),
                        c.get("quantization")))
    for label, model, rev, q in entries:
        if only and label != only:
            continue
        check(label, model, rev, q)
    print(f"\n{len(failures)} FAIL" if failures else "\nall entries OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
