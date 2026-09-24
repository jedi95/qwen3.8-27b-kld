# Qwen3.8-27B quantization KLD — results

Reference: `Qwen/Qwen3.8-27B` BF16 under global `linear_backend=b12x`
(`bf16-b12x`), one dedicated 96 GB Blackwell GPU, TP=1, enforce_eager,
full-vocab teacher-forced prompt logits over the pinned 2048-token
window. KLD = KL(reference || candidate), mean over 2047 scored
positions. Every candidate operand also runs global `b12x`; CUTLASS
rows force the FP4 core GEMM back to FlashInfer CUTLASS (the
measured exception). Engine pin: `configs/environment.lock`.

| candidate (HF model ID) | branch | weights | KLD mean | KLD spread | runs | head KLD | top-1 | kernels |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **BF16 vs itself (runtime floor)** | main | 51.7 GiB | 0.000000 | 0.000000 | 2 | 0.000000 | 1.0000 | runtime floor |
| `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` | `main` | 20.1 GiB | 0.00399 | 0.000000 | 2 | 0.00293 | 0.9761 | exl3 native ext (JIT) |
| `Minachist/Qwen3.8-27B-INT8-AutoRound` | `main-gs128` | 29.3 GiB | 0.00697 | 0.000000 | 2 | 0.00151 | 0.9839 | global `b12x` |
| `turboderp/Qwen3.8-27B-exl3` | `6.00bpw` | 21.4 GiB | 0.00721 | 0.000000 | 2 | 0.00162 | 0.9756 | exl3 native ext (JIT) |
| `lribeiro/Qwen3.8-27B-FP8-Pessoa` | `main` | 27.4 GiB | 0.00787 | 0.000971 | 5 | 0.00187 | 0.9723 | global `b12x` |
| `Minachist/Qwen3.8-27B-INT8-AutoRound` | `main` | 28.8 GiB | 0.00736 | 0.000000 | 2 | 0.00158 | 0.9780 | global `b12x` |
| `turboderp/Qwen3.8-27B-exl3` | `SC_6.00bpw_H6` | 21.4 GiB | 0.00796 | 0.000000 | 2 | 0.00171 | 0.9756 | exl3 native ext (JIT) |
| `turboderp/Qwen3.8-27B-exl3` | `5.00bpw` | 18.5 GiB | 0.00952 | 0.000000 | 2 | 0.00212 | 0.9775 | exl3 native ext (JIT) |
| `Qwen/Qwen3.8-27B-FP8` | `main` | 28.3 GiB | 0.01033 | 0.000000 | 2 | 0.00349 | 0.9629 | global `b12x` |
| `turboderp/Qwen3.8-27B-exl3` | `SC_5.00bpw_H6` | 18.5 GiB | 0.01380 | 0.000000 | 2 | 0.00600 | 0.9668 | exl3 native ext (JIT) |
| `turboderp/Qwen3.8-27B-exl3` | `4.00bpw` | 15.7 GiB | 0.01593 | 0.000000 | 2 | 0.00755 | 0.9619 | exl3 native ext (JIT) |
| `lribeiro/Qwen3.8-27B-Pessoa-5090` | `main` | 20.3 GiB | 0.01634 | 0.000283 | 5 | 0.00693 | 0.9655 | global `b12x` |
| `turboderp/Qwen3.8-27B-exl3` | `SC_4.00bpw_H5` | 15.6 GiB | 0.02140 | 0.000000 | 2 | 0.01141 | 0.9560 | exl3 native ext (JIT) |
| `turboderp/Qwen3.8-27B-exl3` | `3.50bpw` | 14.3 GiB | 0.02301 | 0.000000 | 2 | 0.01187 | 0.9468 | exl3 native ext (JIT) |
| `rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm` | `main` | 22.0 GiB | 0.03447 | 0.000000 | 2 | 0.02018 | 0.9311 | global `b12x`; dynamic-scale FP8 routed to auto (FlashInfer) |
| `unsloth/Qwen3.8-27B-NVFP4` | `main` | 21.0 GiB | 0.03736 | 0.000000 | 2 | 0.02086 | 0.9331 | global `b12x`; FP4 GEMM forced FlashInfer CUTLASS; dynamic-scale FP8 routed to auto (FlashInfer) |
| `unsloth/Qwen3.8-27B-NVFP4` | `main` | 21.0 GiB | 0.04117 | 0.000000 | 2 | 0.02368 | 0.9287 | global `b12x`; dynamic-scale FP8 routed to auto (FlashInfer) |
| `local-inference-lab/Qwen3.8-27B-NVFP4-QAD` | `main` | 23.8 GiB | 0.04296 | 0.000000 | 2 | 0.02596 | 0.9262 | global `b12x`; FP4 GEMM forced FlashInfer CUTLASS |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | `main` | 19.6 GiB | 0.04411 | 0.000000 | 2 | 0.02771 | 0.9331 | global `b12x` |
| `dbirks/Qwen3.8-27B-W4A16-AutoRound` | `main` | 18.1 GiB | 0.04419 | 0.000000 | 2 | 0.02756 | 0.9336 | global `b12x` |
| `turboderp/Qwen3.8-27B-exl3` | `3.00bpw` | 12.9 GiB | 0.04539 | 0.000000 | 2 | 0.02849 | 0.9267 | exl3 native ext (JIT) |
| `local-inference-lab/Qwen3.8-27B-NVFP4-QAD` | `main` | 23.8 GiB | 0.05005 | 0.000000 | 2 | 0.03097 | 0.9150 | global `b12x` |
| `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` | `main` | 22.1 GiB | 0.05025 | 0.000000 | 2 | 0.03167 | 0.9170 | global `b12x` |
| `nvidia/Qwen3.8-27B-NVFP4` | `main` | 20.4 GiB | 0.05275 | 0.000000 | 2 | 0.03043 | 0.9072 | global `b12x`; FP4 GEMM forced FlashInfer CUTLASS |
| `nvidia/Qwen3.8-27B-NVFP4` | `main` | 20.4 GiB | 0.05385 | 0.000000 | 2 | 0.03331 | 0.9145 | global `b12x` |
| `RedHatAI/Qwen3.8-27B-INT4` | `main` | 17.3 GiB | 0.05667 | 0.000000 | 2 | 0.03542 | 0.9179 | global `b12x` |
| `RadixArk/Qwen3.8-27B-NVFP4` | `main` | 20.4 GiB | 0.06479 | 0.000000 | 2 | 0.04127 | 0.8891 | global `b12x` |
| `turboderp/Qwen3.8-27B-exl3` | `SC_3.00bpw_H4` | 12.5 GiB | 0.06948 | 0.000000 | 2 | 0.04464 | 0.9057 | exl3 native ext (JIT) |
| `turboderp/Qwen3.8-27B-exl3` | `2.50bpw` | 11.5 GiB | 0.07465 | 0.000000 | 2 | 0.04537 | 0.9003 | exl3 native ext (JIT) |
| `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` | `main` | 16.7 GiB | 0.09099 | 0.000000 | 2 | 0.05116 | 0.8832 | global `b12x` |
| `turboderp/Qwen3.8-27B-exl3` | `2.00bpw` | 10.0 GiB | 0.16394 | 0.000000 | 2 | 0.10270 | 0.8510 | exl3 native ext (JIT) |
| `turboderp/Qwen3.8-27B-exl3` | `SC_2.00bpw_H3` | 9.5 GiB | 0.22133 | 0.000000 | 2 | 0.12634 | 0.8300 | exl3 native ext (JIT) |

## Deviations & repro notes

- **candidate column = HuggingFace model ID** (capture manifest `model`).
  No operand uses online quantization — the checkpoint weights + their
  bundled config are the source of truth, so there is no separate
  `quant` column. Two rows with the same ID differ only by the FP4 GEMM
  kernel (see `kernels`).
- **branch column = HF revision the capture actually loaded** (manifest
  `revision`; 40-hex SHA pins resolve to `main`, the branch they were
  pinned from). Branch-indexed repos (turboderp exl3 SC_* rungs) are
  distinguished here — the `kernels` column alone does not.
- **weights column = main-model weight bytes on disk** (GiB), measured
  from the safetensors files the model index actually loads in the HF
  cache snapshot (`configs/weight_sizes.json`). Excludes
  tokenizer/config/assets and the separate MTP draft heads (FP8 +0.44,
  unsloth +0.79 GiB) — those are not part of the measured forward pass.
  This is on-disk size, not runtime VRAM (activations/KV/cache differ).
- **b12x BF16 LM-head ≡ cuBLAS**: the b12x `bf16_vocab_projection`
  measured bit-identical to the cuBLAS LM head at N=248044 (KLD 0.0,
  top-1 1.0), so the BF16 reference is kernel-neutral.
- **b12x stack**: `linear_backend="b12x"` globally on every operand.
  b12x covers NVFP4/MXFP8/MXFP4/FP8-scaled linear GEMMs and the BF16
  LM-head vocab projection; it has NO BF16 dense GEMM, so BF16
  MLP/attention linears stay torch/cuBLAS on BOTH the reference and
  candidates (kernel-fair). Layer types without a b12x kernel fall
  back to auto selection per the engine's filter policy — check the
  per-run engine log `Selected/Using ...Kernel` lines for the exact
  per-layer truth.
- **N=96 MXFP8 (GDN in_proj_ba)**: native under b12x (no N>=128 gate,
  unlike FlashInfer).
- **FP4 GEMM exception**: `nvfp4_w4a4=flashinfer_cutlass` control rows
  measure the CUTLASS-vs-b12x rounding delta under an otherwise
  identical b12x stack. Isolated delta (b12x minus CUTLASS):
  unsloth +0.0038, qad +0.0071, nvidia +0.0011 mean KLD — b12x's
  FP4 GEMM is KLD-worse on all three; top-1 falls on unsloth/qad,
  rises on nvidia (0.9072→0.9145). The per-quant override key is
  `nvfp4_w4a4`.
- **Dynamic-scale FP8**: unsloth and prismaqua ship GDN in_proj_qkvz
  as compressed-tensors W8A8-FP8 with dynamic activation scales, which
  the b12x scaled_mm kernel does not support, so those rows route
  `fp8_w8a8: auto` (FlashInfer) while keeping global b12x elsewhere.
- **head KLD**: mean over runs of the KL over the sampler-reachable head of the  reference distribution — the nucleus (top_p=0.95) within top_k=20 of  softmax(z/T), T=1.0, i.e. exactly the deployed reachable set under  Qwen/Qwen3.8-27B generation_config.json defaults (~8.4 vocab entries  on average, max 20), both operands renormalized on that head.  Motivated by QAD-style checkpoints distilled against tokens that can  influence sampled output rather than the full-vocab tail: full-vocab  KLD charges them for tail disagreement the deployed path never  exposes. Computed from the same raw captures (no re-capture) by  `scripts/head_kld.py`, which records the sampling params it used;  floor probe head KLD is 0.0. Same-stack comparability rules apply.
- **EXL3 self-calibration (SC_)**: every SC rung is *worse* than the  corresponding plain rung on this window (6.00: 0.00796 vs 0.00721;  5.00: 0.01380 vs 0.00952; 4.00: 0.02140 vs 0.01593; 3.00: 0.06948 vs  0.04539; 2.00: 0.22133 vs 0.16394), despite similar or identical size.  Treat `SC_` runs as a separate family, not a fidelity upgrade.- **EXL3**: `exllamav3_ext` JIT-built by `scripts/setup_env.sh`;
  `VLLM_EXL3_EXT_PATH` exported by `run_campaign.sh`. EXL3's dequant
  path is kernel-fixed by the exl3 ext; the b12x global backend only
  changes its non-exl3 layers + LM head.
- **Repeat rule**: >=2 fresh-process captures per operand. 35 of
  37 rows are bit-identical across runs (spread 0.000000 — the
  engine is fully deterministic for those checkpoints). The two
  `lribeiro/Qwen3.8-27B-*Pessoa` rows are the measured exception:
  fresh processes disagree at the ~4-8e-4 KLD level (engine
  `VLLM_BATCH_INVARIANT` is unsupported for this GDN_ATTN arch),
  so they run 5x under the identical protocol and report the
  mean; the spread column shows the run-to-run range. The
  reproduction tolerance in `scripts/verify_repro.py`
  (|delta KLD| <= 0.002) covers this spread.
- Repro: `scripts/reproduce.sh` (full) or
  `scripts/run_campaign.sh {ref|<label>|all|report}`;
  `scripts/make_report.py` rebuilds this table.

