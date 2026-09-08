#!/usr/bin/env bash
# Qwen3.8-27B-NVFP4 + DFlash2 — TP4, 1Cat-vLLM 1.5.0, 4× V100-SXM2-32GB
# Target = NVFP4 с lm_head BF16; drafter = incoai BF16 DFlash2.
# Fast path: fp8_e5m2 + FLASH_ATTN_V100 + FULL CUDA Graph + prefix + Mamba align.
# Sampling сервера: T=1.0 top_p=0.95 top_k=20 xhigh. Потолок выхода 64K.
# Eval 16K EOS + seeds 42/123/2026 — только на клиенте (max_tokens/seed).
# Mixed NVFP4: unit-scale patch. 512K = YaRN×2.
# Откат: GPU_UTIL=0.80 BATCHED_TOKENS=4096 ./serve-qwen3.8-nvfp4-dflash4.sh
set -euo pipefail
source ~/bin/1cat-env-15.sh

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SM70_FLASH_ATTN_V100=1
export VLLM_SM70_NVFP4_TURBOMIND=1
cd ~

# --model: dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2
# --speculative-config: incoai/Qwen3.8-27B-DFlash2 @ dedf8df68adfb1afeaf7b7480c0a0243108177b4
MODEL="${MODEL:-/mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2}"
DRAFT="${DRAFT:-/mnt/storage/models/Qwen3.8-27B-DFlash2}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"
MM_IMAGES="${MM_IMAGES:-500}"
# 512K = 524288; натив 262144 → YaRN factor 2
CTX="${CTX:-262144}"
NATIVE_CTX="${NATIVE_CTX:-262144}"
YARN_FACTOR="${YARN_FACTOR:-2.0}"
SEEDS="${SEEDS:-42,123,2026}"
SPEC_TOKENS="${SPEC_TOKENS:-6}"

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
    raise SystemExit(f"cannot patch e5m2 guard in {p}")
PY

if [[ "$DRAFT" == /* ]]; then
  SPEC="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"kv_cache_dtype\":\"auto\",\"num_speculative_tokens\":${SPEC_TOKENS}}"
else
  SPEC="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"revision\":\"dedf8df68adfb1afeaf7b7480c0a0243108177b4\",\"kv_cache_dtype\":\"auto\",\"num_speculative_tokens\":${SPEC_TOKENS}}"
fi

echo "engine: ctx=${CTX} util=${GPU_UTIL} batched=${BATCHED_TOKENS} mm_images=${MM_IMAGES}"
echo "eval sampling (client): seeds=${SEEDS} max_tokens=16384"

ARGS=(
  --model "$MODEL"
  --served-model-name qwen3.8-27b-dflash2
  --trust-remote-code
  --tensor-parallel-size 4
  --attention-backend FLASH_ATTN_V100
  --kv-cache-dtype fp8_e5m2
  --max-model-len "$CTX"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching
  --mamba-cache-mode align
  --enable-chunked-prefill
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --compilation-config '{"cudagraph_mode":"FULL"}'
  --generation-config auto
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"max_new_tokens":65536}'
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
