#!/usr/bin/env python3
"""Compute full-vocab positional KLD between two capture files (contract-compliant).

Per docs/METHODOLOGY.md (adapted from the LIL kld/README.md):
- KL_t(B || Q) = sum_v p_t[v] * (log p_t[v] - log q_t[v]) over the FULL vocabulary
- log_softmax in >= FP32, accumulate summary sums in FP64
- reference-to-candidate direction preserved (asymmetric)
- report negative values beyond tolerance as errors
- publish micro mean/median/p95/p99/p99.9/max, JSD, top-1 agreement

Usage: kld_compare.py <ref.safetensors> <cand.safetensors> [--out summary.json]
                     [--label L] [--ref-manifest M.json] [--cand-manifest M.json]
Files hold fp32 [N, V] log-probabilities (already log_softmax'ed at capture).
When manifests are given, token_first16 identity is enforced (fail-closed).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open

_REPO = Path(__file__).resolve().parent.parent

NEG_TOL = 1e-6  # nats


def load_logits(path: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as h:
        return h.get_tensor("logits")


def positional_kld(ref: torch.Tensor, cand: torch.Tensor, chunk_rows: int = 64):
    """KL(ref || cand) per position, chunked in FP32, accumulated FP64."""
    n, vocab = ref.shape
    assert cand.shape == ref.shape, f"shape mismatch {cand.shape} vs {ref.shape}"
    kld = torch.empty(n, dtype=torch.float64)
    jsd = torch.empty(n, dtype=torch.float64)
    top1_ref = torch.empty(n, dtype=torch.int64)
    top1_cand = torch.empty(n, dtype=torch.int64)
    for s in range(0, n, chunk_rows):
        e = min(n, s + chunk_rows)
        lp = F.log_softmax(ref[s:e].float(), dim=-1)
        lq = F.log_softmax(cand[s:e].float(), dim=-1)
        # F.kl_div(input=lq (candidate log-probs), target=lp (reference log-probs),
        #          log_target=True) = exp(lp) * (lp - lq) = KL(ref || cand). Correct direction.
        kl = F.kl_div(lq, lp, reduction="none", log_target=True).sum(dim=-1)
        # JSD: 0.5*KL(P||M) + 0.5*KL(Q||M), M = (P+Q)/2
        lm = torch.logaddexp(lp, lq) - math.log(2.0)
        p = lp.exp()
        q = lq.exp()
        jsd[s:e] = (0.5 * (p * (lp - lm)).sum(-1) + 0.5 * (q * (lq - lm)).sum(-1)).double()
        kld[s:e] = kl.double()
        top1_ref[s:e] = lp.argmax(dim=-1)
        top1_cand[s:e] = lq.argmax(dim=-1)
    return kld, jsd, top1_ref, top1_cand


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="ref_vs_cand")
    ap.add_argument("--ref-manifest", default=None)
    ap.add_argument("--cand-manifest", default=None)
    args = ap.parse_args()

    for mp in (args.ref_manifest, args.cand_manifest):
        if mp:
            man = json.loads(Path(mp).read_text())
            if "token_first16" not in man:
                raise RuntimeError(f"{mp}: manifest missing token_first16")
    if args.ref_manifest and args.cand_manifest:
        a = json.loads(Path(args.ref_manifest).read_text())["token_first16"]
        b = json.loads(Path(args.cand_manifest).read_text())["token_first16"]
        if a != b:
            raise RuntimeError(
                f"token identity mismatch: ref first16={a} cand first16={b}")

    ref = load_logits(args.ref)
    cand = load_logits(args.cand)
    n, vocab = ref.shape
    print(f"ref {tuple(ref.shape)}  cand {tuple(cand.shape)}", flush=True)

    kld, jsd, t1r, t1c = positional_kld(ref, cand)

    neg = int((kld < -NEG_TOL).sum().item())
    if neg:
        raise RuntimeError(f"{neg} positions with KLD < -{NEG_TOL} nats; refusing to publish")

    k64 = kld.numpy()
    j64 = jsd.numpy()

    summary = {
        "label": args.label,
        # repo-relative so comparison JSONs stay machine-portable
        "ref": _rel(args.ref),
        "cand": _rel(args.cand),
        "positions": int(n),
        "vocab": int(vocab),
        "mean_kld": float(k64.mean()),
        "median_kld": float(torch.median(kld).item()),
        "p95_kld": float(torch.quantile(kld, 0.95).item()),
        "p99_kld": float(torch.quantile(kld, 0.99).item()),
        "p999_kld": float(torch.quantile(kld, 0.999).item()),
        "max_kld": float(k64.max()),
        "mean_jsd": float(j64.mean()),
        "median_jsd": float(torch.median(jsd).item()),
        "top1_agreement": float((t1r == t1c).double().mean().item()),
        "negative_positions": neg,
        "negative_tolerance_nats": NEG_TOL,
    }
    out = json.dumps(summary, indent=2, sort_keys=True)
    print(out, flush=True)
    if args.out:
        Path(args.out).write_text(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
