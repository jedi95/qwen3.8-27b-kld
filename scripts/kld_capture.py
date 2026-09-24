#!/usr/bin/env python3
"""Capture full-vocab prompt logits for one operand (Qwen3.8-27B KLD campaign).

Engine: the pinned vLLM fork from configs/environment.lock (see .env KLD_PY).
Works for BOTH operands:
  - BF16 reference:  --model <hf-id-or-dir>
  - quant candidate: --model <hf-id-or-dir> [--quantization exl3]

Loads the frozen 2048-token window as raw token IDs (never retokenizes),
requests prompt_logprobs=-1 with flat_logprobs=True (this build has no
return_prompt_logits; the densify path is validated by the floor probe),
verifies finiteness, and saves logits_<run>.safetensors = fp32
[context_length-1, vocab] log-probabilities + a manifest.

Ends with os._exit(0) to skip vLLM teardown aborts; the wrapper reaps
orphaned worker PIDs before the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-id", type=int, required=True,
                    help="0 = reference capture (writes manifest.json); "
                         "1 = floor probe repeat; >=2 candidate repeats")
    ap.add_argument("--tokens-file", required=True)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--quantization", default=None, help="e.g. exl3")
    ap.add_argument("--revision", default=None,
                    help="frozen HF revision (branch name or commit hash); "
                         "required for branch-indexed repos (e.g. turboderp exl3)")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--extra-llm-kw", default=None,
                    help="JSON dict merged into LLM(**kwargs), e.g. kernel-config overrides")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token_ids = json.loads(Path(args.tokens_file).read_text())
    assert len(token_ids) == args.context_length, (
        f"expected {args.context_length} tokens, file has {len(token_ids)}")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm as _vllm

    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        max_model_len=args.context_length + 16,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        seed=0,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=-1,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        disable_custom_all_reduce=True,
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.revision:
        kwargs["revision"] = args.revision
    if args.kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.extra_llm_kw:
        extra = json.loads(args.extra_llm_kw)
        kwargs.update(extra)
        manifest_extra = extra
    else:
        manifest_extra = None

    t0 = time.monotonic()
    llm = LLM(**kwargs)
    print(f"model loaded in {time.monotonic() - t0:.0f}s", flush=True)

    prompt: TokensPrompt = {"prompt_token_ids": token_ids}

    supports_rpl = "return_prompt_logits" in inspect.signature(
        SamplingParams.__init__).parameters
    if supports_rpl:
        sp = SamplingParams(prompt_logprobs=1, max_tokens=1, seed=0,
                            return_prompt_logits=True, detokenize=False)
        attr = "prompt_logits"
    else:
        sp = SamplingParams(prompt_logprobs=-1, flat_logprobs=True,
                            max_tokens=1, seed=0, detokenize=False)
        attr = "prompt_logprobs"
    print(f"capture method: {attr}", flush=True)

    t1 = time.monotonic()
    output = llm.generate([prompt], sampling_params=sp)[0]
    print(f"capture took {time.monotonic() - t1:.0f}s", flush=True)

    captured = getattr(output, attr, None)
    if captured is None:
        raise RuntimeError(f"vLLM returned None for {attr}")

    vocab = llm.llm_engine.get_tokenizer().vocab_size
    npos = args.context_length - 1

    if supports_rpl:
        model_logits = captured[:npos, :vocab].detach().cpu().float()
        log_probs = F.log_softmax(model_logits, dim=-1)
    else:
        dense = torch.empty((npos, vocab), dtype=torch.float32)
        pl = captured
        if hasattr(pl, "start_indices"):  # flat logprobs
            for pos in range(npos):
                start = pl.start_indices[pos + 1]
                end = pl.end_indices[pos + 1]
                ids = torch.as_tensor(pl.token_ids[start:end], dtype=torch.long)
                values = torch.as_tensor(pl.logprobs[start:end], dtype=torch.float32)
                row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
                valid = (ids >= 0) & (ids < vocab)
                row[ids[valid]] = values[valid]
                dense[pos] = row
        else:
            for pos in range(npos):
                row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
                entry = pl[pos + 1]
                if entry is not None:
                    for tok, lp in entry.items():
                        tok = int(tok)
                        if 0 <= tok < vocab:
                            row[tok] = float(lp.logprob)
                dense[pos] = row
        log_probs = dense - dense.logsumexp(dim=-1, keepdim=True)

    # Finateness gate: on the flat path, positions where the engine returned
    # FEWER than `vocab` entries mean the capture is sparse top-k, not full
    # vocab — surface the coverage so the floor probe can prove faithfulness.
    if not supports_rpl:
        counts = torch.isfinite(dense).sum(dim=-1)
        cov = counts.min().item()
        print(f"flat coverage: min {cov}/{vocab} entries per position", flush=True)
        if cov < vocab:
            print(f"WARN coverage_gap: min entries {cov} < vocab {vocab}; "
                  "KLD is a lower bound unless -1 returned full vocab", flush=True)

    finite = torch.isfinite(log_probs)
    if not bool(finite.all()):
        raise RuntimeError(
            f"{int((~finite).sum().item())} non-finite logprobs; capture invalid "
            "(aggressive quant producing NaN logits is a real signal — investigate "
            "the checkpoint, not the runner)")

    logits_path = out_dir / f"logits_{args.run_id}.safetensors"
    save_file({"logits": log_probs.contiguous()}, str(logits_path))
    print(f"saved {logits_path} ({logits_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    manifest = {
        "label": args.label,
        "run_id": args.run_id,
        "model": args.model,
        "quantization": args.quantization,
        "revision": args.revision,
        "kv_cache_dtype": args.kv_cache_dtype,
        "dtype": "bfloat16",
        "context_length": args.context_length,
        "scored_positions": npos,
        "vocab_size": vocab,
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_mem_util,
        "vllm_version": _vllm.__version__,
        "logit_shape": list(log_probs.shape),
        "logit_dtype": "float32",
        "logit_key": "logits",
        "logit_sha256": hashlib.sha256(logits_path.read_bytes()).hexdigest(),
        "token_first16": token_ids[:16],
        "token_last16": token_ids[-16:],
        "capture_method": attr,
        "extra_llm_kw": manifest_extra,
        "kld_direction": "KL(reference || candidate)",
        "enforce_eager": True,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 256,
        "seed": 0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / f"manifest_run{args.run_id}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.run_id == 0:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("capture_done " + json.dumps(
        {k: manifest[k] for k in ("label", "run_id", "logit_sha256")}), flush=True)

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
