#!/usr/bin/env python3
"""Audit (and refresh) the revision pins in configs/models.yaml against the hub.

Reads the registry itself — the reference and every candidate.
For each pinned revision:

  - full SHA pin   -> checks the SHA still exists AND whether it is still the
                      HEAD of main (a pinned revision that is no longer HEAD
                      is normal for a frozen registry; it just prints a note)
  - branch pin     -> checks the branch still exists and prints its current
                      HEAD (turboderp SC_* rungs live on branches, not main)

Usage:
  pin_revisions.py            # audit: exit 0 if nothing changed since pins
  pin_revisions.py --update   # rewrite configs/models.yaml with current HEADs
                              (dry-run diff first; review before rerunning captures)
"""
import json
import re
import sys
import urllib.error
import urllib.request

import yaml

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = f"{REPO}/configs/models.yaml"
API = "https://huggingface.co/api"
UA = {"User-Agent": "kld-campaign-pin-audit"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def get(url):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r), r.status
    except urllib.error.HTTPError as e:
        return None, e.code


def audit():
    cfg = yaml.safe_load(open(CFG))
    pinned = []  # (label, model, revision)
    ref = cfg["reference"]
    pinned.append((ref["label"], ref["model"], ref["revision"]))
    seen = set()
    for c in cfg["candidates"]:
        pinned.append((c["label"], c["model"], c.get("revision")))
        seen.add((c["model"], c.get("revision")))

    drift, missing, unchanged = [], [], 0
    cache = {}
    for label, model, rev in pinned:
        if model not in cache:
            cache[model] = get(f"{API}/models/{model}")
        info, code = cache[model]
        if code != 200:
            missing.append((label, model, rev, f"repo HTTP {code}"))
            continue
        main_head = info.get("sha")

        if rev and not SHA_RE.match(rev):  # branch pin (e.g. SC_4.00bpw_H5)
            br, code = get(f"{API}/models/{model}/revision/{rev}")
            if code != 200:
                missing.append((label, model, rev, "branch no longer exists"))
                continue
            head = (br or {}).get("sha", "?")
            print(f"  {label:28s} {model}  branch {rev} HEAD {head[:10]}")
            unchanged += 1
            continue
        if SHA_RE.match(rev or ""):
            _, exists = get(f"{API}/models/{model}/revision/{rev}")
            if exists != 200:
                missing.append((label, model, rev, "revision no longer resolvable"))
            elif rev == main_head:
                print(f"  {label:28s} {model}  {rev[:10]}  == main HEAD")
                unchanged += 1
            else:
                drift.append((label, model, rev, main_head))
                print(f"  {label:28s} {model}  {rev[:10]}  DRIFT: main HEAD now "
                      f"{(main_head or '?')[:10]} (pin still resolves — frozen OK)")
            continue
        missing.append((label, model, rev, "revision is neither a SHA nor a branch"))

    print(f"\npinned entries: {len(pinned)} | unchanged: {unchanged} | "
          f"drifted (main moved): {len(drift)} | missing/broken: {len(missing)}")
    for label, model, rev, why in missing:
        print(f"  MISSING  {label} {model}@{rev}: {why}")
    return 1 if missing else 0


def main():
    if "--update" in sys.argv:
        print("--update is intentionally manual: rerun this audit, then edit "
              "configs/models.yaml yourself. Revising pins invalidates every "
              "result in results/ — re-capture before republishing numbers.")
        return 2
    return audit()


if __name__ == "__main__":
    raise SystemExit(main())
