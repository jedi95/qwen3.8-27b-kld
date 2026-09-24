# KLD Methodology — Qwen3.8-27B quantization campaign

Adapted from the Local Inference Lab full-vocabulary KLD protocol
(Local Inference Lab KLD protocol doc, 2026-09-11 revision; not vendored here) and the
`llm-quantization-benchmarking` skill. Simplified where the source model makes
a branch inapplicable; deviations are called out explicitly.

## 1. Estimand

**End-to-end serving divergence:** `KL(BF16 reference || candidate)` on
teacher-forced, exactly aligned token sequences. Captured logits include the
checkpoint's own LM head — this is the deployed-path number, not a
head-excluded body number.

For reference logits `z_t^B` and candidate logits `z_t^Q` over the full
vocabulary `V`:

```text
p_t = softmax(z_t^B)
q_t = softmax(z_t^Q)
KL_t(B || Q) = Σ_{v∈V} p_t[v] · (log p_t[v] − log q_t[v])   [nats]
```

Comparator contract (enforced in `scripts/kld_compare.py`):

- full-vocabulary comparison — top-k logprobs are never sufficient;
- identical token IDs and position alignment for both operands, verified via
  `manifest.json` `token_first16`;
- `log_softmax` in FP32, summary statistics accumulated in FP64;
- direction preserved (`F.kl_div(input=candidate, target=reference,
  log_target=True)` ⇒ `KL(ref || cand)`);
- non-finite logits or KLD positions abort the run (fail-closed);
- negative KLD beyond 1e-6 nats is an error, never clamped.

Secondary metrics, reported alongside but never replacing forward KLD: JSD and
**top-1 agreement** (fraction of positions where argmax of ref and candidate
match).

## 2. Model fact: dense, no route control

Qwen3.8-27B (`architectures: Qwen3_5ForConditionalGeneration`, `model_type:
qwen3_5`) is a **dense** GDN-hybrid: 64 layers, `linear_attention` × 3 +
`full_attention` every 4th, hidden 5120, vocab 248320, **no `num_experts`** in
`text_config`. The four-cell route-controlled design from the source protocol
does **not** apply; the complete estimand is plain `KL(B||Q)` over the pinned
window. No MoE machinery is imported.

Vision tower: the checkpoint is VL-family. Text-only capture never exercises
`visual.*` logits; we capture through the model's canonical LM output only.
If a quant checkpoint fails to load because of the vision tower, the text-only
config rewrite from the skill (`architectures → Qwen3_5ForCausalLM`, strip
`visual`/`mtp` tensors, keep nested `text_config`) is the sanctioned
workaround — record it as a deviation in the run manifest.

## 3. Serving stack (frozen across all operands)

| Knob | Value |
|---|---|
| Host | see `.env` (`KLD_GPU_IDS`) — one 96 GB Blackwell GPU, exclusive |
| Engine | vLLM fork pinned at `feature/exl3-support` HEAD `51df79b26` (vLLM 0.1.dev21499+g51df79b26, torch 2.13.0+cu130) — see `configs/environment.lock` |
| exl3 support | present in `QUANTIZATION_METHODS` (fork addition) |
| Capture path | in-process `LLM.generate` with prompt logprobs — no HTTP server |
| `dtype` | `bfloat16` for every operand (candidate dequant runs in-engine) |
| `tensor_parallel_size` | **1** (single-GPU pin; frozen across all operands) |
| `enforce_eager` | True |
| `max_num_seqs` | 1 |
| `max_num_batched_tokens` | 256 |
| `enable_prefix_caching` | False |
| `disable_custom_all_reduce` | True |
| `kv_cache_memory_bytes` | 512 MiB |
| `max_model_len` | 2064 (window + 16) |
| `seed` | 0 |
| MTP / speculative decode | **disabled** on every operand (draft heads must not emit logits) |
| KV cache dtype | `auto` (BF16) on every operand |

Engine difference vs other campaigns: this stack is **not** stock vLLM 0.29 —
its repeat floor must be measured fresh (§6) and nothing from prior ladders
on other stacks transfers to it.

## 4. Evaluation data

Frozen window: `data/qwen38-kld-tokens-2048.json`, 2048 token IDs,
sha256 `435403d2…b320`, first-16 IDs pinned in `data/tokens.manifest.json`.
Provenance: a WikiText-2 window reused byte-identical from an earlier
Qwen3.8 quantization benchmark. Captures **load token IDs
directly and never retokenize**.

Current limitation: a single 2048-token literary-prose window,
2047 scored positions. This matches the same-ladder anchors (§8) but is not
the stratified multi-stratum suite the source protocol prescribes. Literary
prose is consistently the *worst* stratum for weight-only quant, so these
numbers are conservative on relative ranking and biased high on absolute KLD.
Multi-stratum expansion (code, math, dialogue, structured) would use
source-document clusters as the sampling unit, with partition membership
frozen before any candidate is measured.

Contamination note: WikiText-2 predates every candidate training cutoff
mentioned on the model cards; no formal overlap scan has been run. Recorded as
unknown-quantifier overlap, per the source protocol's honesty rule.

## 5. Runtime controls & receipts

Per the source protocol, every non-treatment variable is frozen; the treatment
is exactly the checkpoint bytes (+ their declared quantization method). Each
capture writes `captures/<label>/manifest_run<N>.json` recording: model path
and HF revision, quantization method, vLLM version string, tensor shapes and
sha256, token first/last-16, launch kwargs, GPU count, timestamps.
`checksums.txt` is regenerated by `run_campaign.sh`.

Between runs: kill `nvidia-smi --query-compute-apps` PIDs — the capture
runner ends in `os._exit(0)` to skip vLLM teardown aborts, which orphans TP
workers holding VRAM. Never chain two engine inits in one process.

## 6. Runtime floor (acceptance baseline)

The capture path is `prompt_logprobs=-1, flat_logprobs=True` (the densify
path): it returns the **full vocabulary** (248044 entries at every position —
`min == vocab`, no coverage gap) and two fresh-process BF16 captures under
the frozen controls above are **byte-identical
(both `logit_sha256 = e4984335…`), floor = 0.0 nats exactly, top-1 agreement
1.0** (`results/comparisons/floor_probe.json`). Any measured candidate
divergence on this stack is signal, not noise. If the engine stack
changes, re-measure the floor before reading candidate numbers.

No model-independent acceptance bands ("below 0.01 is lossless") are asserted
by this repo. Bands exist only as triage intuition from same-stack history:
floors on other measured stacks ranged from 0.0011 mean (prior LIL fork) to
exactly 0 (stock vLLM 0.29).

## 7. Statistics

Per candidate (from `kld_compare.py`): micro mean, median, p95, p99, p99.9,
max KLD; mean/median JSD; top-1 agreement; positions and vocab asserted;
negative-position count. Each candidate runs N fresh processes (default 2;
registry `runs:`); the ladder reports the mean across runs plus the max−min
spread. 35 of 37 rows are bit-identical across runs (spread 0). Two
checkpoints — `lribeiro/Qwen3.8-27B-FP8-Pessoa` and
`lribeiro/Qwen3.8-27B-Pessoa-5090` — are measurably nondeterministic on this
engine (fresh processes disagree at ~4–8e-4 KLD; their auto-selected fallback
kernels vary run-to-run, and engine `VLLM_BATCH_INVARIANT` is unsupported for
the GDN_ATTN arch), so they run 5x and report the mean; the spread column
makes the noise visible. The reproduction tolerance (|Δ mean_kld| ≤ 0.002)
covers this spread. Bootstrap CIs over source clusters are not computed
(single window ⇒ one cluster ⇒ meaningless).

## 8. Anchors (expectation only — never cross-comparable)

From a community 40-checkpoint MTP-off sweep (unknown stack, direction
unverified) and an earlier EXL3-INT8 benchmark (LIL v20 stack, 2x5090,
teacher-forced 2048-token WikiText prompt KL):

| Checkpoint | Ladder | KLD | Top-1 |
|---|---|---:|---:|
| runtime floor | LIL v20 | 0.0011 | 0.987 |
| EXL3 K4 (4.00 bpw, own quant) | LIL v20 | 0.00170 | — |
| EXL3 K3 (3.00 bpw) | LIL v20 | 0.00753 | — |
| official FP8 baseline | Discord sweep | 0.0133 | 0.9615 |
| NVFP4 GPTQ v14 | Discord sweep | 0.0159 | 0.9567 |
| EXL3 K4 (community) | Discord sweep | 0.0307 | 0.9452 |
| EXL3 K5/K6 | Discord sweep | 0.0082 | 0.9697 |
| mixed INT4 AutoRound | Discord sweep | 0.0609 | 0.9230 |
| AWQ | Discord sweep | 0.0805 | 0.9113 |
| unsloth NVFP4 | Discord sweep | 0.0949 | 0.9054 |

These anchor *ordering expectations only*. Any number from this campaign is
comparable only to other same-stack numbers in this repo.

## 9. Candidate registry

`configs/models.yaml` — frozen list, all cached on the shared local HF hub
(`~/.cache/huggingface/hub`, the staging convention). Community cards' KLD
numbers were **not** consulted for selection beyond registry existence; per the
comparability rule they are unusable for ranking.

| Label | Repo | Method | On-disk |
|---|---|---|---:|
| bf16 (reference) | `Qwen/Qwen3.8-27B` | — | 55.6 GB |
| fp8 | `Qwen/Qwen3.8-27B-FP8` | fp8 | 30.9 GB |
| nvfp4-nvidia | `nvidia/Qwen3.8-27B-NVFP4` | modelopt MIXED_PRECISION | 21.9 GB |
| nvfp4-qad | `local-inference-lab/Qwen3.8-27B-NVFP4-QAD` | modelopt MIXED_PRECISION | 25.6 GB |
| nvfp4-unsloth | `unsloth/Qwen3.8-27B-NVFP4` | compressed-tensors | 23.4 GB |
| awq-int4 | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | compressed-tensors | 21.0 GB |
| w4a16-autoround | `dbirks/Qwen3.8-27B-W4A16-AutoRound` | compressed-tensors | 19.5 GB |
| exl3-k5k6 | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` | exl3 | (hydrated) |

Self-quantized EXL3 K4/K3 ladders (exllamav3 converter, head_bits=8) can be
added with `scripts/run_campaign.sh <label>` after registry entry.

## 10. Known risks / deviations

- **EXL3 self-calibration control.** The turboderp `SC_*` branches are
  compared head-to-head against the plain bpw branches of the same repo
  (`6.00/5.00/4.00/3.00/2.00bpw`). On this window the SC variant is worse at
  *every* rung, with the gap widening as bpw drops (6.00bpw: +11% vs plain →
  2.00bpw: +35%;">the penalty is in the weights, not the runtime: the floor
  probe is 0.0 on both paths). Users choosing EXL3 checkpoints for this model
  should prefer the plain branches or the independently-calibrated
  `malaiwah` K5K6-hydrated build (0.00399 at ~20.1 GiB), which dominates the
  entire 6-bpw turboderp family at fewer bits.
- **flat_logprobs densification** is a sparse→dense reconstruction; tail mass
  outside the returned logprob set becomes −inf and is renormalized. For a
  full-vocab KL this is acceptable only if the engine returns *all* vocab
  entries at `prompt_logprobs=-1` (`max_logprobs=-1`). Validate on the floor
  probe: floor ≈ 0 means the densify path is faithful; a floor matching
  published sparse-top-k underestimates means it is not — in that case the
  campaign switches to patching `return_prompt_logits` from the newer API, or
  the numbers are declared top-k-contaminated and unpublished.
- FP8 checkpoint is the official per-tensor build; card claims are not
  re-verified here (registry entry = artifact identity only).
- If a quant fails to load at TP=2 on this build, record the failure in
  `results/campaign.json` as `load_failed` with the error line — a failed
  load is a finding, not a silent drop.
