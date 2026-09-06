#!/usr/bin/env bash
# Qwen3.8-27B-NVFP4 + DFlash2 — TP4, 1Cat-vLLM 1.5.0 wheel, 4× V100-SXM2-32GB
# Target по умолчанию = QUASAR all-NVFP4; drafter = incoai BF16 DFlash2.
# Релизный профиль: dtype half, fp8_e5m2, util 0.80, batched 4096, seqs 4, probabilistic.
# Mixed: MODEL=/mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2 (unit-scale patch).
# 32GB: GPU_UTIL=0.90 BATCHED_TOKENS=8192
# 512K = YaRN×2. Eval 16K EOS + seeds 42/123/2026 — только на клиенте.
set -euo pipefail
source ~/bin/1cat-env-15.sh

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SM70_FLASH_ATTN_V100=1
export VLLM_SM70_NVFP4_TURBOMIND=1
cd ~

# --model: QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
# --speculative-config: incoai/Qwen3.8-27B-DFlash2 @ dedf8df68adfb1afeaf7b7480c0a0243108177b4
MODEL="${MODEL:-/mnt/storage/models/Qwen3.8-27B-QUASAR-NVFP4}"
DRAFT="${DRAFT:-/mnt/storage/models/Qwen3.8-27B-DFlash2}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.80}"
BATCHED_TOKENS="${BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MM_IMAGES="${MM_IMAGES:-100}"
# 512K = 524288; натив 262144 → YaRN factor 2
CTX="${CTX:-262144}"
NATIVE_CTX="${NATIVE_CTX:-262144}"
YARN_FACTOR="${YARN_FACTOR:-2.0}"
SEEDS="${SEEDS:-42,123,2026}"

[[ -f "$MODEL/config.json" ]] || { echo "No config.json in $MODEL"; exit 1; }
[[ -f "$DRAFT/config.json" ]] || { echo "No config.json in $DRAFT"; exit 1; }
[[ -f "$DRAFT/model.safetensors" ]] || { echo "Drafter weights missing in $DRAFT"; exit 1; }

python - <<'PY'
import torch
from pathlib import Path
from flash_attn_v100 import flash_attn_grouped_verify_max_query_tokens

n = torch.cuda.device_count()
print("GPUs:", n, [torch.cuda.get_device_name(i) for i in range(n)])
if n < 4:
    raise SystemExit(f"need 4 GPUs for TP4, got {n}. CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")
print("DFlash2 grouped verify max Q:", flash_attn_grouped_verify_max_query_tokens())

import vllm
p = Path(vllm.__file__).resolve().parent / "model_executor/layers/attention/attention.py"
text = p.read_text()
old = "if not sm70_flash_v100 or not unit_scale_compatible:"
new = "if not sm70_flash_v100:  # unit e5m2 on mixed NVFP4 (ignore checkpoint KV scales)"
if old in text:
    p.write_text(text.replace(old, new, 1))
    print("patched", p, "for fp8_e5m2 unit-scale")
elif "if not sm70_flash_v100:" in text:
    print("e5m2 unit-scale patch already applied:", p)
else:
    print("warn: e5m2 guard not found in", p, "(QUASAR/wheel may not need the mixed patch)")
PY

if [[ "$DRAFT" == /* ]]; then
  SPEC="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\"}"
else
  SPEC="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"revision\":\"dedf8df68adfb1afeaf7b7480c0a0243108177b4\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\"}"
fi

echo "engine: model=${MODEL} ctx=${CTX} util=${GPU_UTIL} batched=${BATCHED_TOKENS} seqs=${MAX_NUM_SEQS} mm_images=${MM_IMAGES}"
echo "eval sampling (client): seeds=${SEEDS} max_tokens=16384"

ARGS=(
  --model "$MODEL"
  --served-model-name qwen3.8-27b-dflash2
  --trust-remote-code
  --dtype half
  --tensor-parallel-size 4
  --attention-backend FLASH_ATTN_V100
  --kv-cache-dtype fp8_e5m2
  --max-model-len "$CTX"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-prefix-caching
  --mamba-cache-mode align
  --enable-chunked-prefill
  --compilation-config '{"cudagraph_mode":"FULL"}'
  --generation-config auto
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"max_new_tokens":65536}'
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}'
  --speculative-config "$SPEC"
  --host 0.0.0.0
  --port "$PORT"
  --limit-mm-per-prompt "{\"image\":${MM_IMAGES}}"
  --skip-mm-profiling
  --mm-processor-cache-type shm
  --mm-processor-kwargs '{"truncation": false}'
  --mm-encoder-tp-mode data
)

if (( CTX > NATIVE_CTX )); then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  ARGS+=(
    --hf-overrides "{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true, \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"factor\": ${YARN_FACTOR}, \"original_max_position_embeddings\": ${NATIVE_CTX}}}}"
  )
  echo "YaRN factor=${YARN_FACTOR} max-model-len=${CTX} util=${GPU_UTIL}"
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
