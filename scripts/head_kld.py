#!/usr/bin/env python3
"""Add sampler-reachable head KLD columns to comparison JSONs, from existing captures.

Motivation: QAD-style checkpoints are distilled against tokens that can
actually influence sampled output, not the full-vocabulary long tail. The
deployed reachable set is defined by the reference checkpoint's DEFAULT
sampling parameters (generation_config.json), not an arbitrary mass cutoff:

  head_t = nucleus (cumulative prob >= top_p) taken WITHIN the top_k tokens,
           ranked by the temperature-scaled reference distribution
           p_T = softmax(z_ref / T)   (HF semantics: TopKLogitsWarper then
           TopPLogitsWarper)

  p~_t, q~_t = reference / candidate distributions renormalized on head_t
  head_KL_t  = KL(p~_t || q~_t)

The head is the REFERENCE's reachable set because the KL needs one shared
support, and it is the teacher's reachable set that defines deployed output.
Defaults come from Qwen/Qwen3.8-27B generation_config.json
(temperature=1.0, top_k=20, top_p=0.95); the values are recorded in every
output row so a stale column is detectable (--check flags param drift).

Conventions match kld_compare.py: tensors are stored log-probs, log_softmax
FP32, accumulation FP64, chunked, fail-closed on negative KLD beyond
tolerance.

Usage:
  head_kld.py            # fill/refresh head_kld_* into results/comparisons/*.json
  head_kld.py --check    # report rows missing or computed under other params
  head_kld.py --force    # recompute rows that already have current fields

Reads capture paths from each comparison JSON (repo-relative).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open

NEG_TOL = 1e-6
CHUNK = 64

# Reference default sampling params (Qwen/Qwen3.8-27B generation_config.json)
SAMPLE_PARAMS = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}


def load_logits(path: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as h:
        return h.get_tensor("logits")


def head_mask(lp_ref: torch.Tensor, temperature: float,
              top_k: int, top_p: float) -> torch.Tensor:
    """Sampler-reachable support per position: nucleus within top_k of the
    temperature-scaled reference distribution (HF TopK -> TopP semantics)."""
    p = F.log_softmax(lp_ref.float() / temperature, dim=-1).exp()
    sp, idx = p.sort(dim=-1, descending=True)
    cum = sp.cumsum(dim=-1)
    ranks = torch.arange(sp.shape[1], device=sp.device).unsqueeze(0)
    keep = (cum < top_p) | (ranks == 0)   # nucleus, never empty
    keep = keep & (ranks < top_k)         # inside top_k
    mask = torch.zeros_like(keep)
    mask.scatter_(1, idx, keep)
    return mask


def head_kld(ref: torch.Tensor, cand: torch.Tensor, sp: dict):
    """Per-position KL of the renormalized sampler-reachable head.
    Returns (kld, head_size)."""
    n, vocab = ref.shape
    assert cand.shape == ref.shape
    kld = torch.empty(n, dtype=torch.float64)
    size = torch.empty(n, dtype=torch.float64)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        lp = F.log_softmax(ref[s:e].float(), dim=-1)
        lq = F.log_softmax(cand[s:e].float(), dim=-1)
        mask = head_mask(lp, sp["temperature"], sp["top_k"], sp["top_p"])
        p = lp.exp()
        q = lq.exp()
        p_head = (p * mask).sum(dim=-1, keepdim=True).clamp_min(1e-30)
        q_head = (q * mask).sum(dim=-1, keepdim=True).clamp_min(1e-30)
        # contributions only on the head; outside-head entries are forced to
        # exactly zero BEFORE the difference (nan*0 must never enter the sum)
        lp_h = torch.where(mask, lp - torch.log(p_head), torch.zeros_like(lp))
        lq_h = torch.where(mask, lq - torch.log(q_head), torch.zeros_like(lq))
        p_h = p * mask
        kl = (p_h * (lp_h - lq_h)).sum(dim=-1)
        kld[s:e] = kl.double()
        size[s:e] = mask.sum(dim=-1).double()
    return kld, size


def params_stale(d: dict) -> bool:
    return (d.get("head_temperature"), d.get("head_top_k"), d.get("head_top_p")) \
        != (SAMPLE_PARAMS["temperature"], SAMPLE_PARAMS["top_k"], SAMPLE_PARAMS["top_p"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on filename")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    comp_dir = repo / "results" / "comparisons"
    files = sorted(comp_dir.glob("*.json"))
    if args.only:
        files = [f for f in files if args.only in f.name]

    ref_cache: dict[str, torch.Tensor] = {}

    def get(path: Path) -> torch.Tensor:
        if str(path) not in ref_cache:
            ref_cache[str(path)] = load_logits(str(path))
        return ref_cache[str(path)]

    todo, current = [], 0
    for f in files:
        d = json.loads(f.read_text())
        fresh = "head_kld_mean" in d and not params_stale(d) and not args.force
        if args.check:
            if not fresh:
                print(f"STALE {f.name}")
            current += fresh
            continue
        if fresh:
            continue
        todo.append((f, d))

    if args.check:
        print(f"{current}/{len(files)} rows current "
              f"(T={SAMPLE_PARAMS['temperature']}, top_k={SAMPLE_PARAMS['top_k']}, "
              f"top_p={SAMPLE_PARAMS['top_p']})")
        return 0

    for f, d in todo:
        R = get(repo / d["ref"])
        C = get(repo / d["cand"])
        kld, size = head_kld(R, C, SAMPLE_PARAMS)
        neg = int((kld < -NEG_TOL).sum().item())
        if neg:
            raise RuntimeError(f"{f.name}: {neg} negative head-KLD positions; refusing")
        d["head_kld_mean"] = float(kld.mean().item())
        d["head_kld_median"] = float(torch.median(kld).item())
        d["head_kld_max"] = float(kld.max().item())
        d["head_kld_mean_size"] = float(size.mean().item())
        d["head_kld_max_size"] = float(size.max().item())
        d["head_temperature"] = SAMPLE_PARAMS["temperature"]
        d["head_top_k"] = SAMPLE_PARAMS["top_k"]
        d["head_top_p"] = SAMPLE_PARAMS["top_p"]
        d.pop("head_mass", None)   # stale 99%-mass field from v1 of this metric
        f.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
        print(f"{f.name:44s} head_kld={d['head_kld_mean']:.6f} "
              f"(head size {d['head_kld_mean_size']:.1f}, max {d['head_kld_max_size']:.0f})",
              flush=True)
    print(f"wrote {len(todo)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
