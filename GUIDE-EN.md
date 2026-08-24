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
