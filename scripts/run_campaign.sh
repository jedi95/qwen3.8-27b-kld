#!/usr/bin/env bash
# Orchestrator for the Qwen3.8-27B KLD campaign on the benchmark host.
#   run_campaign.sh ref                 -> BF16 reference + floor probe (runs 0,1)
#   run_campaign.sh <label>             -> one candidate (label in configs/models.yaml)
#   run_campaign.sh all                 -> every candidate in registry order
#   run_campaign.sh report              -> rebuild results/campaign.json from comparisons/
# Each engine run is a fresh process; only orphaned workers THIS script spawned
# are reaped (diff against the pre-run PID snapshot — a live vllm.service on
# other GPUs must never be touched).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Machine-local config comes from .env (copy .env.example, or run setup_env.sh).
if [[ ! -f "$REPO/.env" ]]; then
  echo "!! $REPO/.env not found — run scripts/setup_env.sh (or: cp .env.example .env and edit)" >&2
  exit 1
fi
set -a; source "$REPO/.env"; set +a
PY="${KLD_PY:?KLD_PY must point at the venv python (see .env.example)}"
TOK="$REPO/data/qwen38-kld-tokens-2048.json"
CAPS="$REPO/captures"
COMPS="$REPO/results/comparisons"
# Runs on ONE dedicated GPU with TP=1 frozen across ALL operands.
# Pick a free GPU id in .env (KLD_GPU_IDS); do not share it with another
# serving stack mid-run.
export CUDA_VISIBLE_DEVICES="${KLD_GPU_IDS:?KLD_GPU_IDS must be set (see .env.example)}"
TP="${KLD_TP:-1}"
# EXL3 native ext (JIT-built by setup_env.sh; unset lets the engine JIT it)
[[ -n "${VLLM_EXL3_EXT_PATH:-}" ]] && export VLLM_EXL3_EXT_PATH
# GPU-mem fraction: default 0.90 (exclusive GPU)
export KLD_GPU_MEM_UTIL="${KLD_GPU_MEM_UTIL:-0.90}"
mkdir -p "$CAPS" "$COMPS"

reg() { "$PY" "$REPO/scripts/registry.py" "$@"; }

pids_now() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort; }

reap_orphans() { # reap_orphans <before-snapshot-file>
  comm -13 "$1" <(pids_now) | while read -r pid; do
    [[ -n "$pid" ]] || continue
    echo "reaping orphan compute PID $pid (spawned by this campaign)"
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 5
}

capture() { # capture <model> <label> <run-id> [quantization] [extra-llm-kw-json] [revision]
  local model=$1 label=$2 run=$3 quant=${4:-} extra=${5:-} rev=${6:-}
  local qargs=()
  [[ -n "$quant" ]] && qargs=(--quantization "$quant")
  [[ -n "$extra" && "$extra" != "null" ]] && qargs+=(--extra-llm-kw "$extra")
  [[ -n "$rev" && "$rev" != "null" ]] && qargs+=(--revision "$rev")
  local before; before=$(mktemp)
  pids_now > "$before"
  echo "== capture $label run=$run ($model) TP=$TP $(date -u +%FT%TZ)"
  "$PY" "$REPO/scripts/kld_capture.py" \
    --model "$model" --label "$label" --run-id "$run" --output-dir "$CAPS/$label" \
    --tokens-file "$TOK" --context-length 2048 --tp "$TP" \
    --gpu-mem-util "${KLD_GPU_MEM_UTIL:-0.90}" \
    "${qargs[@]}"
  reap_orphans "$before"; rm -f "$before"
}

compare() { # compare <ref-label> <cand-label> <cand-run-id> <out-json> <label>
  "$PY" "$REPO/scripts/kld_compare.py" \
    "$CAPS/$1/logits_0.safetensors" "$CAPS/$2/logits_${3}.safetensors" \
    --ref-manifest "$CAPS/$1/manifest_run0.json" \
    --cand-manifest "$CAPS/$2/manifest_run${3}.json" \
    --out "$4" --label "$5"
}

do_ref() {
  # Reference = BF16 under the global b12x backend (registry `reference`).
  # Floor probe: two fresh-process captures of the identical operand must be
  # bit-identical (docs/METHODOLOGY.md).
  local lbl; lbl=$(reg reference.label)
  capture "$(reg reference.model)" "$lbl" 0 "$(reg reference.quantization)" "$(reg reference.extra_llm_kw)" "$(reg reference.revision)"
  capture "$(reg reference.model)" "$lbl" 1 "$(reg reference.quantization)" "$(reg reference.extra_llm_kw)" "$(reg reference.revision)"
  compare "$lbl" "$lbl" 1 "$COMPS/floor_probe.json" "runtime_floor_KL(run0||run1)"
  echo "runtime floor -> results/comparisons/floor_probe.json"
}

do_candidate() {
  local label=$1
  local spec model quant extra ref
  spec=$(reg candidate "$label")
  model=$(printf '%s' "$spec" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model"])')
  quant=$(printf '%s' "$spec" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["quantization"] or "")')
  extra=$(printf '%s' "$spec" | "$PY" -c 'import json,sys; e=json.load(sys.stdin).get("extra_llm_kw"); print(e or "")')
  ref=$(reg reference.label)
  rev=$(printf '%s' "$spec" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("revision") or "")')
  nruns=$(printf '%s' "$spec" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("runs") or 2)')
  runs=$(seq 2 $((nruns + 1)))
  for run in $runs; do capture "$model" "$label" "$run" "$quant" "$extra" "$rev"; done
  for run in $runs; do
    compare "$ref" "$label" "$run" "$COMPS/${label}_run${run}.json" "${label}_run${run}" \
      || echo "{\"label\": \"${label}_run${run}\", \"error\": \"compare_failed\"}" > "$COMPS/${label}_run${run}.json"
  done
}

do_report() {
  "$PY" - "$COMPS" "$REPO/results/campaign.json" <<'EOF'
import glob, json, os, sys, time
comps, out = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(comps, "*.json"))):
    try:
        rows.append(json.load(open(p)))
    except Exception as e:
        rows.append({"file": os.path.basename(p), "error": f"unparseable: {e}"})
json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "comparisons": rows}, open(out, "w"), indent=2)
print(f"{len(rows)} comparison rows -> {out}")
EOF
}

case "${1:-}" in
  ref) do_ref ;;
  all)
    # Rows the engine cannot load (see METHODOLOGY §10) must not abort the
    # sweep — record the failure and continue.
    for l in $(reg candidates | "$PY" -c 'import json,sys; print(" ".join(json.load(sys.stdin)))'); do
      do_candidate "$l" || {
        echo "!! candidate $l failed; recording load_failed and continuing" >&2
        for run in 2 3; do
          [[ -f "$COMPS/${l}_run${run}.json" ]] || \
            echo "{\"label\": \"${l}_run${run}\", \"error\": \"load_failed\"}" > "$COMPS/${l}_run${run}.json"
        done
      }
    done
    do_report ;;
  report) do_report ;;
  "") echo "usage: run_campaign.sh {ref|<label>|all|report}"; exit 1 ;;
  *) if ! reg candidate "$1" >/dev/null 2>&1; then
       echo "usage: run_campaign.sh {ref|<label>|all|report}  (unknown label: $1)"; exit 1
     fi
     do_candidate "$1" ;;
esac
