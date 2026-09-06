# Installing and running 1Cat-vLLM on Ubuntu 24.04

**Target hardware:** 4× Tesla V100 32GB  
**Version:** 1Cat-vLLM 1.3.0  
**Official references:** [README](https://github.com/1CatAI/1Cat-vLLM) · [Releases](https://github.com/1CatAI/1Cat-vLLM/releases)


## 0. System


| Component    | Version               |
| ------------ | --------------------- |
| OS           | Ubuntu 24.04 LTS      |
| GPU          | 4× V100 32GB          |
| CUDA toolkit | **12.8**              |
| Python       | **3.12**              |
| PyTorch      | cu128 wheels          |
| 1Cat-vLLM    | **1.3.0** (wheel)     |

> CUDA **12.8** is required for 1Cat-vLLM


Check GPUs:

```bash
nvidia-smi
# should show 4× Tesla V100-SXM2-32GB (or PCIe)
```

Also worth checking the cards for ECC errors (double-bit)

```bash
nvidia-smi --query-gpu=index,name,ecc.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv
```
- corrected — recoverable errors (single-bit)
- uncorrected — unrecoverable (usually double-bit) — **this is already bad**
- volatile — since last driver load
- aggregate — cumulative, until you reset

## 1. CUDA 12.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8 build-essential
```

In `~/.bashrc` (or only in the launch script):

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
# for Torch runtime it is better not to keep CUDA lib64 in LD_LIBRARY_PATH permanently

hash -r
nvcc -V   # should show release 12.8
```


## 2. Conda + Python 3.12

```bash
# if miniconda is not installed yet:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n 1cat-vllm python=3.12
conda activate 1cat-vllm
python -m pip install -U pip setuptools wheel
```


## 3. Installing the 1Cat-vLLM wheel

```bash
mkdir -p ~/downloads/1cat && cd ~/downloads/1cat

# download the current wheel from releases:
# https://github.com/1CatAI/1Cat-vLLM/releases/latest
# example filename: 1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl

wget -O 1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl \
  "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.3.0/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl"

python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl
```

The wheel already pulls Torch cu128 and includes `flash_attn_v100` + SM70 kernels.

Verify:

```bash
cd ~
python - <<'PY'
import torch, triton, vllm, sys
import flash_attn_v100
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("flash_attn_v100", flash_attn_v100.__version__)
print("gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

> Check from `~` or `/tmp`, **not** from a repository clone



## (Optional) 4. Disk for models (`/mnt/storage`)

If models live on a separate SSD mounted at `**/mnt/storage**`.  
Weights path: `**/mnt/storage/models/...**`.

### 4.1. Automount on boot

```bash
# UUID of your sda1 (example):
# 46daab66-77b3-489f-ac4c-51ec79715efd

grep -q '/mnt/storage' /etc/fstab || \
  echo 'UUID=46daab66-77b3-489f-ac4c-51ec79715efd  /mnt/storage  ext4  defaults,nofail  0  2' \
  | sudo tee -a /etc/fstab

sudo mkdir -p /mnt/storage
sudo mount -a
sudo chown -R "$USER:$USER" /mnt/storage
mkdir -p /mnt/storage/models

df -h /mnt/storage
findmnt /mnt/storage
```

Optional — a symlink so old `$HOME/models` paths still work:

```bash
ln -sfn /mnt/storage/models ~/models
```

Optional — Hugging Face cache on the same disk:

```bash
mkdir -p /mnt/storage/hf-cache
# once, if you already have a cache on the system NVMe:
# rsync -aH ~/.cache/huggingface/ /mnt/storage/hf-cache/
# mv ~/.cache/huggingface ~/.cache/huggingface.bak
ln -sfn /mnt/storage/hf-cache ~/.cache/huggingface
```

### 4.2 Moving from the system disk

Move already downloaded models from the system disk:

```bash
rsync -aH --info=progress2 ~/models/ /mnt/storage/models/
# then: mv ~/models ~/models.bak && ln -sfn /mnt/storage/models ~/models
```


## 5. Example model download

```bash
# === 27B FP16 (max dense quality; needs ≥60 GiB free) ===
hf download Qwen/Qwen3.8-27B \
  --local-dir /mnt/storage/models/Qwen3.8-27B
```

> if needed: `hf auth login` -> may be required to download Medgemma-27b


After downloading, always check that `config.json` and shard files exist:


```bash
test -f /mnt/storage/models/Qwen3.8-27B/config.json && echo "27B-FP16 OK" || echo "27B-FP16 BROKEN"

du -sh /mnt/storage/models/Qwen3.8-27B
```

If the directory is empty or missing `config.json`, vLLM will raise an error like:

`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/mnt/storage/models/...'


## 6. Launch scripts (4×V100)

```bash
mkdir -p ~/bin ~/logs
```


### 6.1. Shared env — `~/bin/1cat-env.sh`

```bash
#!/usr/bin/env bash
# source ~/bin/1cat-env.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# important: device order = PCI bus
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# do not mix system cuBLAS with the pip Torch build
unset LD_LIBRARY_PATH

# first FlashQLA JIT (if needed) — compiler ≤14
# on 24.04 gcc-13 is usually fine; if problems:
# export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13
# export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"

cd ~   # DO NOT run from a 1Cat-vLLM source checkout
```


### 6.2. Launch script for Qwen3.6-27B FP16 — `~/bin/serve-qwen27b-fp16.sh`

Official weights [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B), no quantization.

The model is **multimodal** (text + vision).

**Script** `~/bin/serve-qwen27b-fp16.sh` (text-only by default — more stable on V100):

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/bin/1cat-env.sh

MODEL="${MODEL:-/mnt/storage/models/Qwen3.8-27B}"
PORT="${PORT:-8000}"
# ENABLE_VISION=1 — image analysis (see below)
ENABLE_VISION="${ENABLE_VISION:-0}"
# ENABLE_YARN=1 — context >256K (factor 2->524K / 4->1M); see YaRN section
ENABLE_YARN="${ENABLE_YARN:-0}"
YARN_FACTOR="${YARN_FACTOR:-2.0}"   # 2.0 or 4.0

[[ -f "$MODEL/config.json" ]] || { echo "No config.json in $MODEL"; exit 1; }

if [[ "$ENABLE_YARN" == "1" ]]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  if [[ -z "${CONTEXT_LENGTH:-}" ]]; then
    case "$YARN_FACTOR" in
      4.0|4) CONTEXT_LENGTH=1010000 ;;
      *)     CONTEXT_LENGTH=524288 ;;
    esac
  fi
else
  # without YaRN: default 128K; native max 262144
  CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
fi

MAX_NUM_SEQS=2
[[ "$ENABLE_YARN" == "1" || "$ENABLE_VISION" == "1" ]] && MAX_NUM_SEQS=1

ARGS=(
  --model "$MODEL"
  --served-model-name qwen3.8-27b
  --trust-remote-code
  --attention-backend FLASH_ATTN_V100
  --tensor-parallel-size 4
  --gpu-memory-utilization 0.90
  --max-model-len "$CONTEXT_LENGTH"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens 8192
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --host 0.0.0.0
  --port "$PORT"
  --dtype half
)

if [[ "$ENABLE_VISION" == "1" ]]; then
  # up to N images per prompt; without this, vision requests are rejected/break
  ARGS+=(
    --limit-mm-per-prompt '{"image":2}'
    --skip-mm-profiling
  )
  # vision uses VRAM — on OOM lower CONTEXT_LENGTH / max-num-seqs
else
  ARGS+=(--language-model-only)
fi

if [[ "$ENABLE_YARN" == "1" ]]; then
  # official Qwen recipe: YaRN via --hf-overrides (--rope-scaling is deprecated)
  ARGS+=(
    --hf-overrides "{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true, \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"factor\": ${YARN_FACTOR}, \"original_max_position_embeddings\": 262144}}}"
  )
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
```

- **`--limit-mm-per-prompt '{"image":5}'`** — limit of multimodal attachments per request: max **5 images**. Without the flag (and without `--language-model-only`) the vLLM default is often stricter/awkward; with `image:5` you can send up to five images in `messages[].content`.
- `--skip-mm-profiling` — skip MM memory profiling at startup (faster start, lower OOM risk on V100).

Make the script executable:

```bash
chmod +x ~/bin/serve-qwen27b-fp16.sh
```

---

A bit more detail on YARN

Native limit of `Qwen3.8-27B`: **262144**. Beyond that — **YaRN** via `--hf-overrides` (the `--rope-scaling` flag **does not work** in recent vLLM).

| `factor` | `--max-model-len` | Window   |
| --------- | ----------------- | -------- |
| `2.0`     | `524288`          | ~524K    |
| `4.0`     | `1010000`         | ~1M      |

**Important**

- YaRN is **static**: it slightly hurts quality on **short** requests. Enable it only when you actually need context **>256K**.
- You need `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, otherwise vLLM will reject `max_model_len` above the derived limit.
- On **4×V100 32GB** FP16 weights are ~54 GiB; KV at 524K/1M often **will not fit**. Realistic: `factor: 2.0` plus a reduced `CONTEXT_LENGTH`, or `--kv-cache-dtype fp8_e5m2`. At 1M you will almost certainly OOM without quantized KV / a shorter length.

```bash
  --kv-cache-dtype fp8_e5m2
```

Do not enable `--calculate-kv-scales` unless you explicitly need it. On MiniMax you can try it if there is not enough KV for the context.

---

## 7. Launch

```bash

# text-only (default, native ≤256K):
~/bin/serve-qwen27b-fp16.sh

# vision (images):
ENABLE_VISION=1 CONTEXT_LENGTH=65536 ~/bin/serve-qwen27b-fp16.sh

# background (vision):
# ENABLE_VISION=1 nohup ~/bin/serve-qwen27b-fp16.sh > ~/logs/1cat-27b-fp16-vl.log 2>&1 &

# native 256K text-only:
# CONTEXT_LENGTH=262144 ~/bin/serve-qwen27b-fp16.sh
# on OOM: CONTEXT_LENGTH=65536 ~/bin/serve-qwen27b-fp16.sh
```


Check the API (name = `--served-model-name`):

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one word: capital of France?"}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Image example (the server must be started with `ENABLE_VISION=1`):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the image briefly."},
        {"type": "image_url", "image_url": {"url": "https://www.example.com/demo.jpg"}}
      ]
    }],
    "max_completion_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

From a local image file

```bash
IMG=./pic5.jpg
B64=$(base64 -i "$IMG" | tr -d '\n')

# check that the string is not empty:
echo ${#B64}

curl http://192.168.1.114:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d "$(jq -n \
    --arg model "qwen3.8-27b" \
    --arg b64 "$B64" \
    '{
      model: $model,
      messages: [{
        role: "user",
        content: [
          {type: "text", text: "Describe the image briefly."},
          {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $b64)}}
        ]
      }],
      max_completion_tokens: 128,
      chat_template_kwargs: {enable_thinking: false}
    }')"
```


## (Optional) 8. Autostart — systemd

> Replace **YOUR_USER** with your username

File `/etc/systemd/system/1cat-vllm.service`:

```ini
[Unit]
Description=1Cat-vLLM OpenAI API
After=network.target nvidia-persistenced.service mnt-storage.mount
RequiresMountsFor=/mnt/storage

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3
Environment=CUDA_HOME=/usr/local/cuda-12.8
Environment=PATH=/usr/local/cuda-12.8/bin:/home/YOUR_USER/miniconda3/envs/1cat-vllm-1.2.2/bin:/usr/bin
ExecStart=/home/YOUR_USER/bin/serve-qwen27b-awq.sh
Restart=on-failure
RestartSec=10
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now 1cat-vllm
sudo journalctl -u 1cat-vllm -f
```

---

## Common errors


### `HFValidationError: Repo id must be in the form...`

The model directory is missing, empty, or has no `config.json`. Re-download the model and check the files.

### `CUBLAS_STATUS_INVALID_VALUE`

Mixing system CUDA libraries with pip Torch. Run `unset LD_LIBRARY_PATH` before launch. Do not put `/usr/local/cuda-12.8/lib64` in `LD_LIBRARY_PATH` for serve runtime.

### Import from source instead of the wheel

Launch from a directory with a `1Cat-vLLM` clone. Switch to `~` or `/tmp`.

### GPU busy / Free memory too low

Stop other processes (vLLM zombies, ComfyUI on the same cards). `nvidia-smi` → `kill -9` leftover PIDs.

---

# Part 2. llama.cpp — alternative: Q8_0 GGUF, MTP, 500K context

> This part is fully independent from the vLLM sections: no conda/vLLM needed, just CUDA 12.8 (section 1) and build tools.
>
> Use case: squeeze the maximum context out of 4×V100 (Q8 weights + quantized KV) and/or use **MTP** (Multi-Token Prediction) — speculative decoding through the model's built-in MTP head.
>
> ⚠ llama.cpp must support your specific model (and MTP) — use a recent release, see [Releases](https://github.com/ggml-org/llama.cpp/releases).

## 9. Building llama.cpp (V100 = SM70)

```bash
sudo apt install -y git cmake

git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
git fetch --tags
git checkout <tag-of-current-release>   # pin a release, not a moving master

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --config Release -j"$(nproc)"
```

- **`-DCMAKE_CUDA_ARCHITECTURES=70`** — compile for V100 (SM70) only. Without the flag CMake auto-detects the GPU via `nvidia-smi` and builds the same thing, but pinning it explicitly is insurance against "built for the wrong hardware".
- **Flash attention does not work on V100** — ggml-cuda FA kernels target SM80+. Do not force `-fa on`: the default attention path is perfectly fine; the build log may note that FA is unavailable for this arch (that's normal).

Install the binaries:

```bash
sudo install -D -m755 build/bin/llama-* /usr/local/bin/
llama-server --version    # should show cuda: yes
```

## 10. Weights: HF → GGUF Q8_0

### 10.1. Converter environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n gguf-tools python=3.12
conda activate gguf-tools
python -m pip install -U pip
python -m pip install gguf sentencepiece protobuf tokenizers safetensors
```

### 10.2. Convert and quantize

```bash
mkdir -p /mnt/storage/models/gguf
cd ~/llama.cpp   # the converter uses relative imports from the repo

# 1) HF → f16 GGUF (intermediate file, ~55 GB)
python convert_hf_to_gguf.py /mnt/storage/models/Qwen3.8-27B \
  --outtype f16 \
  --outfile /mnt/storage/models/gguf/Qwen3.8-27B-f16.gguf

# 2) f16 → Q8_0 (final file, ~28–29 GB)
# positional args: input output type  (there is no --output flag)
llama-quantize \
  /mnt/storage/models/gguf/Qwen3.8-27B-f16.gguf \
  /mnt/storage/models/gguf/Qwen3.8-27B-Q8_0.gguf \
  Q8_0

du -sh /mnt/storage/models/gguf/*
```

- **MTP**: if the HF weights contain MTP tensors (`mtp.*`), the converter saves them into the GGUF — llama.cpp detects MTP automatically at startup (look for MTP/draft lines in the log). If there are none, the GGUF was made with an old converter — rebuild it with a newer llama.cpp.
- Alternative: skip conversion and download a ready-made Q8_0 GGUF (with MTP) from the Hugging Face community.

## 11. Launch script — `~/bin/serve-qwen27b-llamacpp.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SPLIT_MODE="${SPLIT_MODE:-layer}"            # none|layer|row|tensor
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL="${MODEL:-/mnt/storage/models/gguf/Qwen3.8-27B-Q8_0.gguf}"
PORT="${PORT:-8001}"
# 500K context: YaRN ×2 relative to the native 262144
CTX="${CTX:-524280}"
NATIVE_CTX="${NATIVE_CTX:-262144}"
# MTP: how many draft tokens per step (0 = disabled)
DRAFT_MAX="${DRAFT_MAX:-3}"
# KV cache quantization: required for large context
KV_TYPE="${KV_TYPE:-q8_0}"
# the script sets -ngl and -c itself → --fit on conflicts and aborts
FIT="${FIT:-off}"

[[ -f "$MODEL" ]] || { echo "No model file: $MODEL"; exit 1; }

ARGS=(
  --model "$MODEL"
  --alias qwen3.8-27b        # model name in the API
  --host 0.0.0.0
  --port "$PORT"
  -ngl 99                    # all layers on GPU
  -c "$CTX"                  # context size
  -sm "$SPLIT_MODE"
  --fit "$FIT"
  -np 1                      # 1 slot: concurrency does not fit at 500K
  -b 2048 -ub 512            # prompt batch / ubatch
  --cache-type-k "$KV_TYPE"  # KV quant (K)
  --cache-type-v "$KV_TYPE"  # KV quant (V)
  --jinja                     # chat template from the GGUF (tool calls, thinking)
  --cont-batching
)

# context above the native window — YaRN (same approach as the vLLM part)
if [[ "$CTX" -gt "$NATIVE_CTX" ]]; then
  ARGS+=(
    --rope-scaling yarn
    --rope-scale 2.0
    --rope-freq-base 10000000
    --yarn-orig-ctx "$NATIVE_CTX"
  )
fi

# MTP: the head is in the same GGUF; a second file is not needed
if [[ "$DRAFT_MAX" -gt 0 ]]; then
  ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$DRAFT_MAX")
fi

exec llama-server "${ARGS[@]}"
```

```bash
chmod +x ~/bin/serve-qwen27b-llamacpp.sh
```

```bash
# target config — all 4×V100, 500K:
./serve-qwen27b-llamacpp.sh

# 2 GPUs: 500K will not fit. Start with a smaller context:
CUDA_VISIBLE_DEVICES=0,1 CTX=65536 ./serve-qwen27b-llamacpp.sh

# tensor-parallel: needs Flash Attention (SM80+), does not work on V100.
# CUDA_VISIBLE_DEVICES=0,1 SPLIT_MODE=tensor KV_TYPE=f16 CTX=65536 ./serve-qwen27b-llamacpp.sh
```

### About the 500K context

- Native window of `Qwen3.8-27B` = **262144**. Beyond that — **YaRN** (factor 2 → ~524K), the same trick as in the vLLM part. The YaRN flags mirror the model's `rope_parameters` (`rope_theta = 1e7`, native 262144); for a different model, substitute its values.
- **The KV budget is what decides whether it fits:**

  `KV per token = 2 × n_layers × n_kv_heads × head_dim × bytes(KV type)`

  `llama-server` prints the real size in the startup log (`KV self size (MB)`). At 500K with `q8_0` KV this is tens of GB; with `f16` KV, 500K **will not fit** next to Q8 weights in 4×32 GB. On OOM: reduce `CTX` or set `KV_TYPE=q4_0` (quality degrades noticeably).

### About MTP

- MTP = the model predicts several tokens per step through its built-in MTP head. Current llama.cpp: `--spec-type draft-mtp --spec-draft-n-max N` (no second GGUF). The old `--draft` / `--draft-max` flags **have been removed**.
- `N` — how many draft tokens (0–4). On V100, **3** is usually the sweet spot: watch the acceptance rate and speed in the log. If the gain is near zero, try `DRAFT_MAX=1` or `0`.
- MTP mostly speeds up **generation**; prefill is unaffected.

## 12. Launch and verification

```bash
# 500K + MTP (target configuration):
~/bin/serve-qwen27b-llamacpp.sh

# same without MTP (for speed comparison):
DRAFT_MAX=0 ~/bin/serve-qwen27b-llamacpp.sh

# native 256K, no YaRN:
CTX=262144 ~/bin/serve-qwen27b-llamacpp.sh

# background:
nohup ~/bin/serve-qwen27b-llamacpp.sh > ~/logs/llamacpp-q8-500k.log 2>&1 &
tail -f ~/logs/llamacpp-q8-500k.log
```

Check the API (name = `--alias`):

```bash
curl -s http://127.0.0.1:8001/v1/models | jq .

curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one word: capital of France?"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

> `llama-server`'s API is OpenAI-compatible; the `Authorization` header is optional (you can issue a key via `--api-key`).

For autostart, take the unit from section 8, renaming the service and setting `ExecStart=/home/YOUR_USER/bin/serve-qwen27b-llamacpp.sh`.

## 13. Optimizations and common issues (llama.cpp)

| Task | How |
| ---- | --- |
| Does not fit (KV OOM) | `CTX=65536 …`, `KV_TYPE=q4_0`, `-np 1` |
| Concurrency at small context | separate script: `-np 4 -c 32768` |
| MTP gives no gain | `DRAFT_MAX=1`; check MTP is in the log; rebuild the GGUF |
| Tool calls break | `--jinja` (already in the script) |
| Slow prefill at 500K | expected: prefill is linear in context; `-b 2048 -ub 512` |
| MoE models (if needed) | `--n-cpu-moe N` — move experts to CPU |
| Not enough RAM | `--mmap` (default); `--mlock` to pin weights in RAM |

**Typical errors**

- **`n_gpu_layers already set by user to 99, abort`** — `--fit on` (llama.cpp default) wants to lower `-ngl`, but the script already pinned it. Use `--fit off` (the script sets this). If you then get CUDA OOM, you are out of VRAM: on **2×V100** drop `CTX=524280` and use `CTX=65536` (or smaller), or run on **4 GPUs**.
- **`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`** — `--split-mode tensor` cannot auto-fit memory. On **V100**, `tensor` will almost certainly fail later: it needs Flash Attention (SM80+) and unquantized KV (`f16`/`bf16`). For V100 use `SPLIT_MODE=layer`.
- **`invalid argument: --rope-frequency-base`** — current llama.cpp uses `--rope-freq-base`; original context is `--yarn-orig-ctx`, not `--rope-scale-original`.
- **`the argument has been removed` / `invalid argument: --draft`** — MTP is now `--spec-type draft-mtp --spec-draft-n-max N`.
- **No MTP/draft mentions in the log** — the MTP layers are not in the GGUF. Rebuild the GGUF (section 10.2) with the new converter.
- **`CUDA error` at startup** — binaries were not built for SM70. Check `-DCMAKE_CUDA_ARCHITECTURES=70` and rebuild.
- **OOM on weights** — you launched the f16 GGUF (~55 GB) instead of Q8_0.
- **`-fa on` does nothing / errors** — normal for V100 (SM70), see section 9.

---

# Part 3. 3D pipeline: ComfyUI (reference images) + Hunyuan3D 2.1 (image → GLB)

> The whole server picture (4×V100 32GB, `192.168.1.114`):
>
> | GPU | Service | Port | For |
> | --- | ------- | ---- | --- |
> | 0–1 | llama.cpp + Qwen3.8-27B (Part 2) | 8001\* | LLM: text, long context, MTP |
> | 2 | ComfyUI: SDXL + FLUX.1-schnell | 8188 | reference images |
> | 3 | Hunyuan3D 2.1 (FastAPI) | 8090 | image → textured GLB |
>
> \* `ai-status.sh` (section 16) checks llama.cpp on `:8080` — fix the line in the script to your port, or run llama.cpp with `PORT=8080`.
>
> **Pipeline:** the LLM describes the object → ComfyUI renders a reference image → Hunyuan3D builds a 3D model with textures → output is a GLB.
>
> ⚠ In this layout llama.cpp is left on **2×V100** (0–1): a 500K context **does not fit** on two cards (Part 2, section 11) — run `CUDA_VISIBLE_DEVICES=0,1 CTX=65536 ~/bin/serve-qwen27b-llamacpp.sh`, or plan a restart on 4 GPUs.

## 14. What the script installs and why

| Component | Goes to | Why | Size |
| --------- | ------- | --- | ---- |
| HuggingFace CLI (`hf` / `huggingface-cli`) | `~/bin/hf-cli-venv` (only if not in PATH) | downloading models from Hugging Face; handles gated repos via `HF_TOKEN` | — |
| **SDXL base 1.0** (checkpoint) | `/mnt/storage/models/comfyui/checkpoints/` | base image model for ComfyUI — **required** | ~6.9 GB |
| **SDXL VAE** | `/mnt/storage/models/comfyui/vae/` | VAE (image decode) for SDXL — **required** | ~335 MB |
| **FLUX.1-schnell** (checkpoint) | `…/comfyui/checkpoints/` | higher-quality images (distilled, fast) — optional, gated | ~23.8 GB |
| **FLUX VAE** (`ae.safetensors`) | `…/comfyui/vae/` | VAE for FLUX — optional | ~335 MB |
| **T5-XXL (fp16)** | `…/comfyui/clip/t5xxl_fp16.safetensors` | main FLUX text encoder. **Strictly fp16**: V100 (SM70) has no fp8 — optional | ~9.9 GB |
| **CLIP-L** | `…/comfyui/clip/clip_l.safetensors` | auxiliary FLUX text encoder — optional | ~250 MB |
| **ComfyUI** (app) | `~/apps/ComfyUI` — **clone it beforehand**; the script only checks for it | node-based image generation: Web UI + API on `:8188` | — |
| `comfy-models.json` | `~/apps/ComfyUI/comfy-models.json` | points ComfyUI at the extra model root `/mnt/storage/models/comfyui` (the `checkpoints/ vae/ clip/` layout matches, so nothing is moved) | — |
| **Hunyuan3D-2.1** (app) | `~/apps/Hunyuan3D-2.1` (git clone) | image → textured 3D model (shape + paint), FastAPI on `:8090` | ~1 GB |
| Hunyuan3D weights | `/mnt/storage/models/Hunyuan3D-2.1` + symlink `~/apps/Hunyuan3D-2.1/weights` | shape + paint model weights | ~20 GB |
| **torch 2.5.1 (cu124)** + `requirements.txt` | `~/apps/Hunyuan3D-2.1/venv` (Python 3.10/3.11) | the officially validated Hunyuan3D configuration. Not 3.12: the pins (numpy 1.24.4, bpy 4.0, pymeshlab 2022.2) do not build | ~2.5 GB |
| `custom_rasterizer` | same venv (C++/CUDA extension) | differentiable rasterizer for texture painting | — |
| `mesh_inpaint_processor` | `~/apps/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/` | texture inpainting on the mesh (pybind11). Compiled with the **venv python**: their `compile_mesh_painter.sh` calls the system `python3-config`, so the `.so` would get the wrong interpreter's suffix | — |
| **RealESRGAN_x4plus** | `~/apps/Hunyuan3D-2.1/hy3dpaint/ckpt/` | 4× upscaling of generated textures | ~64 MB |
| `~/bin/comfyui-gpu2.sh` | `~/bin/` | launch ComfyUI on GPU 2, port 8188 | — |
| `~/bin/hunyuan3d-gpu3.sh` | `~/bin/` | launch Hunyuan3D on GPU 3, port 8090 | — |
| `~/bin/ai-status.sh` | `~/bin/` | `nvidia-smi` + liveness of `:8188` / `:8090` / `:8080` (llama.cpp) | — |
| systemd units (optional) | `~/bin/comfyui-gpu2.service`, `~/bin/hunyuan3d-gpu3.service` | autostart after reboot | — |

**Disk total:** ~28 GB required (SDXL + VAE + Hunyuan3D weights), ~62 GB with the FLUX stack. The script checks for ~60 GB free on `/mnt/storage`.

**Licenses:** SDXL — Stability Community License (commercial use OK); FLUX.1-schnell — Apache 2.0, **but the HF repo is gated**: you must accept the license on the repo page + provide `HF_TOKEN`.

## 15. Prerequisites (before running the script)

1. **ComfyUI already cloned** — the script does not clone it; without the directory it dies:

   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI ~/apps/ComfyUI
   ```

2. **`~/bin/comfy-env.sh`** — the ComfyUI environment (the launch script sources it). Example — a separate conda env with torch:

   ```bash
   cat > ~/bin/comfy-env.sh <<'EOF'
   #!/usr/bin/env bash
   # env for ComfyUI — adjust to your own environment
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate comfyui
   cd ~/apps/ComfyUI
   EOF
   ```

3. **Compiler** for Hunyuan3D's CUDA extensions:

   ```bash
   sudo apt install -y build-essential
   ```

4. **Python 3.10/3.11 — optional**: the script looks for `python3.10`/`python3.11` itself; if absent it installs [uv](https://astral.sh/uv) (no root) and fetches CPython 3.10.
5. **~60 GB free** on `/mnt/storage` (section 4).

## 16. The `setup-3d-pipeline.sh` script

Save it as `setup-3d-pipeline.sh` (e.g. in `~/`) and run:

```bash
bash setup-3d-pipeline.sh
```

The script is **idempotent**: a re-run skips finished steps (downloads/installs) and only fetches what is missing.

```bash
#!/usr/bin/env bash
#
# setup-3d-pipeline.sh — server arkalaust-AI (4x NVIDIA V100 32GB, 192.168.1.114)
#
# GPU layout:
#   GPU 0-1   llama.cpp + Qwen3.8-27B            (already installed)
#   GPU 2     ComfyUI: SDXL + FLUX.1-schnell     (reference images)
#   GPU 3     Hunyuan3D 2.1 FastAPI              (image -> textured GLB)
#
# Run:  bash setup-3d-pipeline.sh
# Idempotent: a re-run skips finished steps (downloads/installs).
#
# Everything is downloaded to /mnt/storage/models, services go to ~/apps.
#
set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/mnt/storage/models}"
COMFY_DIR="${COMFY_DIR:-$HOME/apps/ComfyUI}"
HY3D_DIR="${HY3D_DIR:-$HOME/apps/Hunyuan3D-2.1}"
HY3D_WEIGHTS="${HY3D_WEIGHTS:-$MODELS_ROOT/Hunyuan3D-2.1}"
COMFY_MODELS="${COMFY_MODELS:-$MODELS_ROOT/comfyui}"
BIN_DIR="$HOME/bin"

mkdir -p "$BIN_DIR"

log()  { echo -e "\n\033[1;36m==== $* ====\033[0m"; }
ok()   { echo -e "\033[1;32m[ ok ]\033[0m $*"; }
skip() { echo -e "\033[1;33m[skip]\033[0m $*"; }
die()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

# HuggingFace CLI: prefer the new `hf` (huggingface-cli is deprecated),
# fall back to the old one; if neither exists — install it ourselves.
HFCLI_VENV="$BIN_DIR/hf-cli-venv"
if command -v hf >/dev/null 2>&1; then
  HFCLI=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HFCLI=(huggingface-cli)
else
  echo "  HuggingFace CLI not found — installing"
  if python3 -m pip install --user -q "huggingface_hub[cli]" 2>/dev/null \
     || pip3 install --user -q "huggingface_hub[cli]" 2>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    # PEP 668 (Ubuntu 24.04+): the system python does not allow pip --user —
    # put it in a separate venv, leave the system untouched.
    echo "  pip --user unavailable (PEP 668) — creating venv $HFCLI_VENV"
    python3 -m venv "$HFCLI_VENV" \
      || die "could not create a venv (need the python3-venv package: sudo apt install python3.12-venv)"
    "$HFCLI_VENV/bin/pip" install -q "huggingface_hub[cli]"
  fi
  if command -v hf >/dev/null 2>&1; then
    HFCLI=(hf)
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HFCLI=(huggingface-cli)
  elif [[ -x "$HFCLI_VENV/bin/hf" ]]; then
    HFCLI=("$HFCLI_VENV/bin/hf")
  elif [[ -x "$HFCLI_VENV/bin/huggingface-cli" ]]; then
    HFCLI=("$HFCLI_VENV/bin/huggingface-cli")
  else
    die "could not install the HuggingFace CLI"
  fi
fi
command -v "${HFCLI[0]}" >/dev/null 2>&1 || die "HuggingFace CLI not found in PATH: ${HFCLI[0]}"

# dl_file <repo> <file> <dest_dir> [required=1|0]
# Download via huggingface-cli (works with gated repos when HF_TOKEN is set).
dl_file() {
  local repo="$1" file="$2" dest_dir="$3" required="${4:-1}"
  local dest="$dest_dir/$file"
  if [[ -s "$dest" ]]; then
    skip "already there: ${dest#"$MODELS_ROOT"/}"
    return 0
  fi
  mkdir -p "$dest_dir"
  echo "  downloading: $repo / $file"
  if HF_TOKEN="${HF_TOKEN:-}" "${HFCLI[@]}" download "$repo" "$file" --local-dir "$dest_dir"; then
    ok "saved: ${dest#"$MODELS_ROOT"/} ($(du -h "$dest" | cut -f1))"
  elif [[ "$required" == "1" ]]; then
    die "error: $repo / $file (gated repos need HF_TOKEN: huggingface.co/settings/tokens)"
  else
    echo -e "  \033[1;33m[warn]\033[0m skipped (optional, HF_TOKEN required): $file"
  fi
}

# ------------------------------------------------------------------
log "0/5 Preflight"
command -v git     >/dev/null || die "git not found"
command -v curl    >/dev/null || die "curl not found"
command -v python3 >/dev/null || die "python3 not found"
PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "  python3 (system): $PYVER — for Hunyuan3D, step 3 will pick 3.10/3.11 automatically"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "  free on $MODELS_ROOT: $(df -BG --output=avail "$MODELS_ROOT" | tail -1 | tr -d ' ' ) GB (~60 needed)"

# ------------------------------------------------------------------
log "1/5 Image models for ComfyUI -> $COMFY_MODELS"
# Licenses: SDXL — Stability Community (commercial OK), FLUX.1-schnell — Apache 2.0
# (but the HF repo is gated: license checkbox accepted + HF_TOKEN required).
# t5xxl strictly fp16 — V100 does not support fp8!
# SDXL is the required part; the FLUX stack is optional (required=0):
# without a token it is skipped with a warning and the pipeline still works.
dl_file "stabilityai/stable-diffusion-xl-base-1.0" "sd_xl_base_1.0.safetensors" "$COMFY_MODELS/checkpoints"
dl_file "stabilityai/sdxl-vae"                     "sdxl_vae.safetensors"       "$COMFY_MODELS/vae"
dl_file "black-forest-labs/FLUX.1-schnell"         "flux1-schnell.safetensors"  "$COMFY_MODELS/checkpoints" 0
dl_file "black-forest-labs/FLUX.1-schnell"         "ae.safetensors"             "$COMFY_MODELS/vae" 0
dl_file "comfyanonymous/flux_text_encoders"        "t5xxl_fp16.safetensors"     "$COMFY_MODELS/clip" 0
dl_file "comfyanonymous/flux_text_encoders"        "clip_l.safetensors"         "$COMFY_MODELS/clip" 0
ok "image models ready (SDXL required; FLUX optional)"

# ------------------------------------------------------------------
log "2/5 ComfyUI: model paths + launch script on GPU 2"
[[ -d "$COMFY_DIR" ]] || die "ComfyUI not found: $COMFY_DIR (override COMFY_DIR=...)"

# ComfyUI picks up $COMFY_MODELS as an extra model root
# (the checkpoints/ vae/ clip/ layout matches) — nothing is moved.
cat > "$COMFY_DIR/comfy-models.json" <<EOF
{ "base_path": "$COMFY_MODELS" }
EOF
ok "extra model paths: $COMFY_DIR/comfy-models.json"

cat > "$BIN_DIR/comfyui-gpu2.sh" <<'EOF'
#!/usr/bin/env bash
# ComfyUI on GPU 2 — image generation, API on :8188
source ~/bin/comfy-env.sh
exec python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --cuda-device 2 \
  --extra-model-paths-config "$HOME/apps/ComfyUI/comfy-models.json"
EOF
chmod +x "$BIN_DIR/comfyui-gpu2.sh"
ok "$BIN_DIR/comfyui-gpu2.sh"

# ------------------------------------------------------------------
log "3/5 Hunyuan3D 2.1: repo + venv + weights"
if [[ ! -d "$HY3D_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HY3D_DIR"
  ok "cloned: $HY3D_DIR"
else
  skip "repo present: $HY3D_DIR"
fi

# --- Python: the repo is validated on 3.10; the pins (numpy 1.24.4, bpy 4.0,
#     pymeshlab 2022.2) do not build on 3.12. Look for 3.10/3.11, otherwise
#     fetch CPython 3.10 via uv (no root).
PYBIN=""
for c in python3.10 python3.11; do
  if command -v "$c" >/dev/null 2>&1; then PYBIN="$c"; break; fi
done
USE_UV=""
if [[ -z "$PYBIN" ]]; then
  echo "  python3.10/3.11 not in PATH — installing uv to fetch CPython 3.10 (no root)"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || die "could not install uv (https://astral.sh/uv)"
  USE_UV=1
fi

# a venv with a different python (e.g. 3.12 from a previous run) — recreate it
if [[ -x "$HY3D_DIR/venv/bin/python" ]]; then
  VENV_PYVER="$("$HY3D_DIR/venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_PYVER" != "3.10" && "$VENV_PYVER" != "3.11" ]]; then
    echo "  current venv is python $VENV_PYVER (need 3.10/3.11) — recreating"
    rm -rf "$HY3D_DIR/venv"
  fi
fi

if [[ ! -d "$HY3D_DIR/venv" ]]; then
  if [[ -n "$USE_UV" ]]; then
    uv venv --python 3.10 "$HY3D_DIR/venv"
  else
    "$PYBIN" -m venv "$HY3D_DIR/venv"
  fi
  ok "venv created (python: $("$HY3D_DIR/venv/bin/python" --version 2>&1))"
else
  skip "venv present"
fi

VPY="$HY3D_DIR/venv/bin/python"
vpip() {
  if [[ -x "$HY3D_DIR/venv/bin/pip" ]]; then
    "$HY3D_DIR/venv/bin/pip" "$@"
  else
    uv pip --python "$VPY" "$@"
  fi
}

if [[ ! -f "$HY3D_DIR/venv/.deps-installed" ]]; then
  command -v g++ >/dev/null || die "g++ not found (needed for the CUDA extensions): sudo apt install build-essential"
  # 1) torch cu124 — the officially validated configuration
  vpip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  # 2) the rest from requirements
  vpip install -r "$HY3D_DIR/requirements.txt"
  # 3) custom C++/CUDA paint extension: custom_rasterizer
  ( cd "$HY3D_DIR/hy3dpaint/custom_rasterizer" && vpip install -e . )
  # 4) mesh painter. Their compile_mesh_painter.sh calls the system python3-config,
  #    so the .so gets the wrong python's suffix — run the same command,
  #    but with the venv python (headers + suffix from the same interpreter).
  ( cd "$HY3D_DIR/hy3dpaint/DifferentiableRenderer" \
    && c++ -O3 -Wall -shared -std=c++11 -fPIC $($VPY -m pybind11 --includes) \
         mesh_inpaint_processor.cpp \
         -o "mesh_inpaint_processor$($VPY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" )
  # 5) RealESRGAN checkpoint (texture upscale)
  mkdir -p "$HY3D_DIR/hy3dpaint/ckpt"
  if [[ ! -s "$HY3D_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]]; then
    curl -L -C - -o "$HY3D_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
  fi
  touch "$HY3D_DIR/venv/.deps-installed"
  ok "dependencies and extensions installed"
else
  skip "dependencies already installed (rebuild: rm $HY3D_DIR/venv/.deps-installed)"
fi

mkdir -p "$HY3D_WEIGHTS"
if [[ -z "$(ls -A "$HY3D_WEIGHTS" 2>/dev/null)" ]]; then
  # requirements pins huggingface-hub==0.30.2 — the venv already has its own huggingface-cli
  "$HY3D_DIR/venv/bin/huggingface-cli" download tencent/Hunyuan3D-2.1 --local-dir "$HY3D_WEIGHTS" \
    || die "Hunyuan3D weights failed to download (if the repo is gated — a token is needed: hf auth login)"
  ok "weights downloaded (~20 GB) -> $HY3D_WEIGHTS"
else
  skip "weights already present: $HY3D_WEIGHTS"
fi

# the repo looks for the weights in ./weights
ln -sfn "$HY3D_WEIGHTS" "$HY3D_DIR/weights"
ok "symlink: $HY3D_DIR/weights -> $HY3D_WEIGHTS"

cat > "$BIN_DIR/hunyuan3d-gpu3.sh" <<'EOF'
#!/usr/bin/env bash
# Hunyuan3D 2.1 FastAPI on GPU 3 — image -> textured GLB, API on :8090
cd ~/apps/Hunyuan3D-2.1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=3
exec python fastapi_server.py --host 0.0.0.0 --port 8090 --enable_tex
EOF
chmod +x "$BIN_DIR/hunyuan3d-gpu3.sh"
ok "$BIN_DIR/hunyuan3d-gpu3.sh"

# ------------------------------------------------------------------
log "4/5 systemd units (autostart — optional)"
gen_unit() { # <name> <description> <script>
  cat > "$BIN_DIR/$1.service" <<EOF
[Unit]
Description=$2
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
ExecStart=$BIN_DIR/$3
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
}
gen_unit comfyui-gpu2   "ComfyUI on GPU 2 (port 8188)"        "comfyui-gpu2.sh"
gen_unit hunyuan3d-gpu3 "Hunyuan3D 2.1 on GPU 3 (port 8090)"  "hunyuan3d-gpu3.sh"
ok "units: $BIN_DIR/comfyui-gpu2.service, $BIN_DIR/hunyuan3d-gpu3.service"
echo "  Enable autostart (optional):"
echo "    sudo cp ~/bin/comfyui-gpu2.service ~/bin/hunyuan3d-gpu3.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable --now comfyui-gpu2 hunyuan3d-gpu3"

# ------------------------------------------------------------------
log "5/5 Status script"
cat > "$BIN_DIR/ai-status.sh" <<'EOF'
#!/usr/bin/env bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo
curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null && echo "ComfyUI    :8188  OK" || echo "ComfyUI    :8188  down"
curl -s -m 3 http://127.0.0.1:8090/docs           >/dev/null && echo "Hunyuan3D  :8090  OK" || echo "Hunyuan3D  :8090  down"
curl -s -m 3 http://127.0.0.1:8080/health         >/dev/null && echo "llama.cpp  :8080  OK" || echo "llama.cpp  :8080  down (different port? fix it here)"
EOF
chmod +x "$BIN_DIR/ai-status.sh"
ok "$BIN_DIR/ai-status.sh"

# ------------------------------------------------------------------
log "Done"
cat <<EOF

Next:
  1) ~/bin/comfyui-gpu2.sh      # or: systemctl start comfyui-gpu2
  2) ~/bin/hunyuan3d-gpu3.sh    # or: systemctl start hunyuan3d-gpu3
  3) ~/bin/ai-status.sh         # check GPUs and ports

Check from the Mac side (the ports must be reachable over LAN):
  curl http://192.168.1.114:8188/system_stats
  curl http://192.168.1.114:8090/docs

If Hunyuan3D's fastapi_server.py does not know the --enable_tex flag — check
  python fastapi_server.py --help
and fix ~/bin/hunyuan3d-gpu3.sh.

FLUX.1-schnell skipped (gated repo)? To fetch it later:
  1) huggingface.co — create an account and a token (settings/tokens, read is enough)
  2) accept the license on the black-forest-labs/FLUX.1-schnell page
  3) HF_TOKEN=hf_xxxx bash setup-3d-pipeline.sh   # fetches only the FLUX stack
EOF
```

## 17. Launch, status, autostart

```bash
~/bin/comfyui-gpu2.sh      # ComfyUI on GPU 2 — http://192.168.1.114:8188 (Web UI + API)
~/bin/hunyuan3d-gpu3.sh    # Hunyuan3D on GPU 3 — http://192.168.1.114:8090 (Swagger: /docs)
~/bin/ai-status.sh         # nvidia-smi + liveness of :8188 / :8090 / :8080
```

Check from the Mac side (the ports must be reachable over LAN):

```bash
curl http://192.168.1.114:8188/system_stats
curl http://192.168.1.114:8090/docs
```

Autostart after reboot (optional; the script has already generated the units):

```bash
sudo cp ~/bin/comfyui-gpu2.service ~/bin/hunyuan3d-gpu3.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now comfyui-gpu2 hunyuan3d-gpu3
```

**Notes**

- `--enable_tex` in Hunyuan3D enables the full "shape + texture" pipeline; if your `fastapi_server.py` version does not know the flag, run `python fastapi_server.py --help` and fix `~/bin/hunyuan3d-gpu3.sh` (the flags change between releases).
- `ai-status.sh` checks llama.cpp on `:8080`, while Part 2 starts it on `8001` by default — either fix the line in `ai-status.sh`, or run llama.cpp with `PORT=8080`.
- FLUX stack skipped (no `HF_TOKEN`)? To fetch it later: accept the license on [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and run `HF_TOKEN=hf_xxxx bash setup-3d-pipeline.sh` — only it will be downloaded.
- In this part's layout llama.cpp lives on **2 GPUs** (0–1): a 500K context does not fit there, see the warning at the top of Part 3.
