#!/usr/bin/env python3
"""Build results/RESULTS.md from results/comparisons/*.json + manifests.

Stack: global linear_backend=b12x for every operand; the reference is BF16
under the same b12x backend (`bf16-b12x`). Rows: one per candidate (mean
KLD + top-1 over its captures; see Repeat rule) plus the runtime floor. CUTLASS FP4
control rows carry a linear_backend_per_quant nvfp4_w4a4=flashinfer_cutlass
override.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPS = REPO / "results" / "comparisons"
CAPS = REPO / "captures"


def kld_of(path):
    d = json.load(open(path))
    return (d["mean_kld"], d["median_kld"], d["top1_agreement"],
            d.get("head_kld_mean"))


def manifest(label, run):
    p = CAPS / label / f"manifest_run{run}.json"
    return json.load(open(p)) if p.exists() else {}


SIZES_PATH = REPO / "configs" / "weight_sizes.json"


def weight_sizes():
    """HF model ID -> main-weight bytes on disk (measured from the HF cache
    snapshot; MTP head files excluded — see file's _note)."""
    if not SIZES_PATH.exists():
        return {}
    return json.load(open(SIZES_PATH)).get("sizes", {})


def fmt5(v):
    return "—" if v is None else f"{v:.5f}"


def fmt6(v):
    return "—" if v is None else f"{v:.6f}"


def fmt_size(key_hf, key_label, sizes, revision=None):
    s = (sizes.get(f"{key_hf}@{revision}") if revision else None) \
        or sizes.get(key_label) or sizes.get(key_hf)
    if not s:
        return "—"
    return f"{s['main_bytes'] / 1024**3:.1f} GiB"


def branch_for(m):
    """Branch name for the table. Manifest `revision` is either a 40-hex
    commit (every SHA-pinned row was pinned from refs/main) or a branch name
    (branch-indexed repos like turboderp exl3). Absent = main."""
    rev = m.get("revision")
    if not rev:
        return "main"
    if len(rev) == 40 and all(c in "0123456789abcdef" for c in rev):
        return "main"
    return rev


def describe_kernel(m):
    """(quant-label, deviation-note) from a run manifest."""
    quant = m.get("quantization")
    extra = m.get("extra_llm_kw") or {}
    kc = (extra.get("kernel_config") or {})
    lpq = kc.get("linear_backend_per_quant") or {}

    q = quant or "auto-detected"
    dev = "global `b12x`"
    if quant == "exl3":
        dev = "exl3 native ext (JIT)"
    if lpq.get("nvfp4_w4a4") == "flashinfer_cutlass":
        dev += "; FP4 GEMM forced FlashInfer CUTLASS"
    if lpq.get("fp8_w8a8") == "auto":
        dev += "; dynamic-scale FP8 routed to auto (FlashInfer)"
    return q, dev


def main():
    labels = sorted({p.stem.rsplit("_run", 1)[0] for p in COMPS.glob("*.json")
                     if not p.stem.startswith("floor")})
    lines = ["# Qwen3.8-27B quantization KLD — results",
             "",
             "Reference: `Qwen/Qwen3.8-27B` BF16 under global `linear_backend=b12x`",
             "(`bf16-b12x`), one dedicated 96 GB Blackwell GPU, TP=1, enforce_eager,",
             "full-vocab teacher-forced prompt logits over the pinned 2048-token",
             "window. KLD = KL(reference || candidate), mean over 2047 scored",
             "positions. Every candidate operand also runs global `b12x`; CUTLASS",
             "rows force the FP4 core GEMM back to FlashInfer CUTLASS (the",
             "measured exception). Engine pin: `configs/environment.lock`.",
             "",
             "| candidate (HF model ID) | branch | weights | KLD mean | KLD spread | runs | head KLD | top-1 | kernels |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]

    p = COMPS / "floor_probe.json"
    if p.exists():
        f = json.load(open(p))
        lines.append(f"| **BF16 vs itself (runtime floor)** | main | "
                     f"{fmt_size('Qwen/Qwen3.8-27B', None, weight_sizes())} | {f['mean_kld']:.6f} | "
                     f"0.000000 | 2 | {f.get('head_kld_mean', float('nan')):.6f} | "
                     f"{f['top1_agreement']:.4f} | runtime floor |")

    def _runs(lab):
        out = []
        for p in sorted(COMPS.glob(f"{lab}_run*.json")):
            try:
                out.append(json.load(open(p)))
            except Exception:
                pass
        return out
    labels.sort(key=lambda l: (min((r["mean_kld"] for r in _runs(l)), default=9e9)))

    for lab in labels:
        m2 = manifest(lab, 2)
        _, dev = describe_kernel(m2)
        hf = (m2 or {}).get("model") or lab
        rev = (m2 or {}).get("revision")
        sz = fmt_size(hf, lab, weight_sizes(), rev)
        rr = _runs(lab)
        kl = [r["mean_kld"] for r in rr]
        t1 = [r["top1_agreement"] for r in rr]
        hd = [r["head_kld_mean"] for r in rr if r.get("head_kld_mean") is not None]
        if not kl:
            kl, t1, hd = [None], [None], []
        lines.append(
            f"| `{hf}` | `{branch_for(m2)}` | {sz} | {fmt5(sum(kl)/len(kl) if all(v is not None for v in kl) else None)} | "
            f"{fmt6((max(kl) - min(kl)) if all(v is not None for v in kl) and len(kl) > 1 else (0.0 if len(kl) == 1 else None))} | "
            f"{len(rr)} | {fmt5(sum(hd)/len(hd) if hd else None)} | "
            f"{(f'{sum(t1)/len(t1):.4f}' if all(v is not None for v in t1) else '—')} | {dev} |")

    lines += ["",
              "## Deviations & repro notes",
              "",
              "- **candidate column = HuggingFace model ID** (capture manifest `model`).",
              "  No operand uses online quantization — the checkpoint weights + their",
              "  bundled config are the source of truth, so there is no separate",
              "  `quant` column. Two rows with the same ID differ only by the FP4 GEMM",
              "  kernel (see `kernels`).",
              "- **branch column = HF revision the capture actually loaded** (manifest",
              "  `revision`; 40-hex SHA pins resolve to `main`, the branch they were",
              "  pinned from). Branch-indexed repos (turboderp exl3 SC_* rungs) are",
              "  distinguished here — the `kernels` column alone does not.",
              "- **weights column = main-model weight bytes on disk** (GiB), measured",
              "  from the safetensors files the model index actually loads in the HF",
              "  cache snapshot (`configs/weight_sizes.json`). Excludes",
              "  tokenizer/config/assets and the separate MTP draft heads (FP8 +0.44,",
              "  unsloth +0.79 GiB) — those are not part of the measured forward pass.",
              "  This is on-disk size, not runtime VRAM (activations/KV/cache differ).",
              "- **b12x BF16 LM-head ≡ cuBLAS**: the b12x `bf16_vocab_projection`",
              "  measured bit-identical to the cuBLAS LM head at N=248044 (KLD 0.0,",
              "  top-1 1.0), so the BF16 reference is kernel-neutral.",
              "- **b12x stack**: `linear_backend=\"b12x\"` globally on every operand.",
              "  b12x covers NVFP4/MXFP8/MXFP4/FP8-scaled linear GEMMs and the BF16",
              "  LM-head vocab projection; it has NO BF16 dense GEMM, so BF16",
              "  MLP/attention linears stay torch/cuBLAS on BOTH the reference and",
              "  candidates (kernel-fair). Layer types without a b12x kernel fall",
              "  back to auto selection per the engine's filter policy — check the",
              "  per-run engine log `Selected/Using ...Kernel` lines for the exact",
              "  per-layer truth.",
              "- **N=96 MXFP8 (GDN in_proj_ba)**: native under b12x (no N>=128 gate,",
              "  unlike FlashInfer).",
              "- **FP4 GEMM exception**: `nvfp4_w4a4=flashinfer_cutlass` control rows",
              "  measure the CUTLASS-vs-b12x rounding delta under an otherwise",
              "  identical b12x stack. Isolated delta (b12x minus CUTLASS):",
              "  unsloth +0.0038, qad +0.0071, nvidia +0.0011 mean KLD — b12x's",
              "  FP4 GEMM is KLD-worse on all three; top-1 falls on unsloth/qad,",
              "  rises on nvidia (0.9072→0.9145). The per-quant override key is",
              "  `nvfp4_w4a4`.",
              "- **Dynamic-scale FP8**: unsloth and prismaqua ship GDN in_proj_qkvz",
              "  as compressed-tensors W8A8-FP8 with dynamic activation scales, which",
              "  the b12x scaled_mm kernel does not support, so those rows route",
              "  `fp8_w8a8: auto` (FlashInfer) while keeping global b12x elsewhere.",
              "- **head KLD**: mean over runs of the KL over the sampler-reachable head of the"
              "  reference distribution — the nucleus (top_p=0.95) within top_k=20 of"
              "  softmax(z/T), T=1.0, i.e. exactly the deployed reachable set under"
              "  Qwen/Qwen3.8-27B generation_config.json defaults (~8.4 vocab entries"
              "  on average, max 20), both operands renormalized on that head."
              "  Motivated by QAD-style checkpoints distilled against tokens that can"
              "  influence sampled output rather than the full-vocab tail: full-vocab"
              "  KLD charges them for tail disagreement the deployed path never"
              "  exposes. Computed from the same raw captures (no re-capture) by"
              "  `scripts/head_kld.py`, which records the sampling params it used;"
              "  floor probe head KLD is 0.0. Same-stack comparability rules apply.",
              "- **EXL3 self-calibration (SC_)**: every SC rung is *worse* than the"
              "  corresponding plain rung on this window (6.00: 0.00796 vs 0.00721;"
              "  5.00: 0.01380 vs 0.00952; 4.00: 0.02140 vs 0.01593; 3.00: 0.06948 vs"
              "  0.04539; 2.00: 0.22133 vs 0.16394), despite similar or identical size."
              "  Treat `SC_` runs as a separate family, not a fidelity upgrade."
              "- **EXL3**: `exllamav3_ext` JIT-built by `scripts/setup_env.sh`;",
              "  `VLLM_EXL3_EXT_PATH` exported by `run_campaign.sh`. EXL3's dequant",
              "  path is kernel-fixed by the exl3 ext; the b12x global backend only",
              "  changes its non-exl3 layers + LM head.",
              "- **Repeat rule**: >=2 fresh-process captures per operand. 35 of",
              "  37 rows are bit-identical across runs (spread 0.000000 — the",
              "  engine is fully deterministic for those checkpoints). The two",
              "  `lribeiro/Qwen3.8-27B-*Pessoa` rows are the measured exception:",
              "  fresh processes disagree at the ~4-8e-4 KLD level (engine",
              "  `VLLM_BATCH_INVARIANT` is unsupported for this GDN_ATTN arch),",
              "  so they run 5x under the identical protocol and report the",
              "  mean; the spread column shows the run-to-run range. The",
              "  reproduction tolerance in `scripts/verify_repro.py`",
              "  (|delta KLD| <= 0.002) covers this spread.",
              "- Repro: `scripts/reproduce.sh` (full) or",
              "  `scripts/run_campaign.sh {ref|<label>|all|report}`;",
              "  `scripts/make_report.py` rebuilds this table.",
              ""]
    out = REPO / "results" / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    unstable = []
    for lab in labels:
        kl = [r["mean_kld"] for r in _runs(lab)]
        if len(kl) > 1 and (max(kl) - min(kl)) > 1e-6:
            unstable.append(f"{lab} (spread {max(kl)-min(kl):.6f})")
    if unstable:
        print(f"NOTE KLD spread >1e-6 (multi-run mean rows): {unstable}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
