#!/usr/bin/env bash
# =============================================================================
# Qwen3.8-27B-NVFP4 (QUASAR) + DFlash2 — 1Cat-vLLM 1.5.0, 4× V100-SXM2-32GB
# =============================================================================
# Стек:  1CatAI/1Cat-vLLM v1.5.0 (NVFP4 на SM70: PR #228, #445), wheel,
#        conda 1cat-vllm-15, torch 2.10.0+cu128, CUDA 12.8.
# База:  «tested four-V100 production profile» из RELEASE.md 1.5.0 +
#        fast 32GB uplift из GUIDE-NVFP4-DFLASH2-RU.md §3
#        (GPU_UTIL=0.90, BATCHED_TOKENS=8192) + контракт 16K-гейта качества
#        (T=1.0, top_p=0.95, top_k=20, xhigh, full CUDA Graph, prefix cache,
#        Mamba align; DFlash2 PPL-дельта vs target ≈ +0.00005, PR #346).
#
# Скорость: DFlash2 — 7 черновиков (checkpoint-native width), probabilistic
#        rejection sampling, grouped q8 verify по E5M2 KV; FULL CUDA Graph;
#        chunked prefill; prefix cache + Mamba align; mm-encoder data-TP.
# Качество: native 262K (без YaRN), thinking xhigh, E5M2 target KV.
#
# Изображения: до 100 на промпт (--limit-mm-per-prompt).
#
# Откат (OOM/нестабильность) — релизный gate 1.5.0:
#   GPU_UTIL=0.80 BATCHED_TOKENS=4096 MAX_NUM_SEQS=4 \
#     ./serve-qwen3.8-nvfp4-dflash2-fast.sh
#   (в guide §5 vision числится фактором риска OOM; при OOM на картинках
#    также уберите --skip-mm-profiling — он экономит время старта ценой
#    точного резерва под encoder)
#
# Прочие ручки:
#   CTX=524288            -> YaRN×2 (экстраполяция RoPE, медленнее)
#   TEMPERATURE=0.6       -> precise-coding профиль (README: MBPP 29/31,
#                            decode ~245 tok/s; опционально, не глобальный дефолт)
#   REASONING_EFFORT=high -> быстрее ответы, меньше thinking
#   MODEL=/mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2 -> mixed NVFP4
#                            (dfischermittwald, lm_head BF16) — скрипт сам
#                            применит e5m2 unit-scale патч к attention.py
#
# Первый старт долгий: AOT graph-cache на SM70 отключён по умолчанию
# (иначе детерминированный дрейф токенов). Прогрейте сервис перед
# боевым трафиком. Не enforce-eager.
# =============================================================================
set -euo pipefail
source ~/bin/1cat-env-15.sh

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SM70_FLASH_ATTN_V100=1
export VLLM_SM70_NVFP4_TURBOMIND=1
cd ~

# QUASAR — recommended fully-quantized target 1.5.0 (~19.7 GiB, PR #445).
# Альтернатива mixed: /mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2
MODEL="${MODEL:-/mnt/storage/models/Qwen3.8-27B-QUASAR-NVFP4}"
DRAFT="${DRAFT:-/mnt/storage/models/Qwen3.8-27B-DFlash2}"   # BF16, не квантовать
PORT="${PORT:-8000}"

# ---- fast-профиль для 32GB (релизный gate: 0.80 / 4096 / 4) ----
GPU_UTIL="${GPU_UTIL:-0.90}"
BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
CTX="${CTX:-262144}"                     # native 256K — максимум качества (без YaRN)
NATIVE_CTX="${NATIVE_CTX:-262144}"
YARN_FACTOR="${YARN_FACTOR:-2.0}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"          # checkpoint-native width (block 8)
MM_IMAGES="${MM_IMAGES:-100}"            # изображений на промпт (контракт)
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
TEMPERATURE="${TEMPERATURE:-1.0}"        # рецепт 1Cat/Qwen thinking; 0.6 = код

[[ -f "$MODEL/config.json" ]] || { echo "No config.json in $MODEL"; exit 1; }
[[ -f "$DRAFT/config.json" ]] || { echo "No config.json in $DRAFT"; exit 1; }
[[ -f "$DRAFT/model.safetensors" ]] || { echo "Drafter weights missing in $DRAFT"; exit 1; }

python - <<'PY'
import torch
from flash_attn_v100 import flash_attn_grouped_verify_max_query_tokens

n = torch.cuda.device_count()
print("GPUs:", n, [torch.cuda.get_device_name(i) for i in range(n)])
if n < 4:
    raise SystemExit(
        f"need 4 GPUs for TP4, got {n}. "
        f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}"
    )
print("DFlash2 grouped verify max Q:", flash_attn_grouped_verify_max_query_tokens())
PY

# e5m2 unit-scale патч — только для mixed NVFP4 (dfischermittwald, lm_head BF16).
# QUASAR — full NVFP4 без checkpoint KV scales: release-gate #445 проходит
# на свежем wheel без патча (иначе были бы потеряны реальные KV scales).
case "${MODEL##*/}" in
  *QUASAR*) ;;
  *)
    python - <<'PY'
from pathlib import Path
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
    ;;
esac

# ДРАФТЕР: pin ревизии для HF-id; для локального пути ревизия не нужна.
if [[ "$DRAFT" == /* ]]; then
  SPEC="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"num_speculative_tokens\":${SPEC_TOKENS}}"
else
  SPEC="{\"method\":\"dflash\",\"model\":\"${DRAFT}\",\"revision\":\"dedf8df68adfb1afeaf7b7480c0a0243108177b4\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"num_speculative_tokens\":${SPEC_TOKENS}}"
fi

echo "engine: model=${MODEL##*/} ctx=${CTX} util=${GPU_UTIL} batched=${BATCHED_TOKENS} seqs=${MAX_NUM_SEQS} spec=${SPEC_TOKENS} mm_images=${MM_IMAGES} thinking=${REASONING_EFFORT} T=${TEMPERATURE}"

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
  --override-generation-config "{\"temperature\":${TEMPERATURE},\"top_p\":0.95,\"top_k\":20,\"max_new_tokens\":65536}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --default-chat-template-kwargs "{\"enable_thinking\":true,\"reasoning_effort\":\"${REASONING_EFFORT}\"}"
  --speculative-config "$SPEC"
  --host 0.0.0.0
  --port "$PORT"
  --limit-mm-per-prompt "{\"image\":${MM_IMAGES}}"
  --skip-mm-profiling
  --mm-processor-cache-type shm
  --mm-processor-kwargs '{"truncation": false}'
  --mm-encoder-tp-mode data
)

# 512K = 524288: YaRN×2 поверх native 262144.
if (( CTX > NATIVE_CTX )); then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  ARGS+=(
    --hf-overrides "{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true, \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"factor\": ${YARN_FACTOR}, \"original_max_position_embeddings\": ${NATIVE_CTX}}}}"
  )
  echo "YaRN factor=${YARN_FACTOR} max-model-len=${CTX} util=${GPU_UTIL}"
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
