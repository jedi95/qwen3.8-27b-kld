#!/usr/bin/env python3
"""Registry helper for run_campaign.sh — prints frozen candidate info as JSON.

  registry.py reference.model
  registry.py candidates          # JSON list
  registry.py candidate <label>   # JSON {model, quantization}
"""
import json
import sys

import yaml

REPO = __file__.rsplit("/", 2)[0]


def main() -> int:
    cfg = yaml.safe_load(open(f"{REPO}/configs/models.yaml"))
    what = sys.argv[1]
    if what == "candidates":
        print(json.dumps([c["label"] for c in cfg["candidates"]]))
    elif what == "candidate":
        for c in cfg["candidates"]:
            if c["label"] == sys.argv[2]:
                q = c.get("quantization")
                print(json.dumps({"model": c["model"],
                                  "quantization": None if q in (None, "null") else q,
                                  "runs": c.get("runs", 2),
                                  "revision": c.get("revision"),
                                  "display": c.get("display"),
                                  "extra_llm_kw": c.get("extra_llm_kw"),
                                  "ref": c.get("ref", "bf16")}))
                return 0
        sys.exit(f"label {sys.argv[2]} not in registry")
    else:  # dotted path
        node = cfg
        for part in what.split("."):
            node = node[part]
        print("" if node is None else node)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
