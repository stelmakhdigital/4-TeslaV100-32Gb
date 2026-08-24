# Установка и запуск 1Cat-vLLM на Ubuntu 24.04

**Целевое железо:** 4× Tesla V100 32GB  
**Версия:** 1Cat-vLLM 1.3.0  
**Официальные референсы:** [README](https://github.com/1CatAI/1Cat-vLLM) · [Releases](https://github.com/1CatAI/1Cat-vLLM/releases)


## 0. Система


| Компонент    | Версия                |
| ------------ | --------------------- |
| OS           | Ubuntu 24.04 LTS      |
| GPU          | 4× V100 32GB          |
| CUDA toolkit | **12.8**              |
| Python       | **3.12**              |
| PyTorch      | cu128 wheels          |
| 1Cat-vLLM    | **1.3.0** (wheel)     |

> CUDA **12.8** требуется для работы с 1Cat-vLLM


Проверка GPU:

```bash
nvidia-smi
# покажет 4× Tesla V100-SXM2-32GB (или PCIe)
```

Желательно еще проверить карты на ECC ошибки (double-bit)

```bash
nvidia-smi --query-gpu=index,name,ecc.mode.current,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv
```
- corrected - исправимые ошибки (single-bit)
- uncorrected - неисправимые (обычно double-bit) — **это уже плохо**
- volatile - с последней загрузки драйвера
- aggregate - накопительно, пока не сбросите

## 1. CUDA 12.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8 build-essential
```

В `~/.bashrc` (или только в скрипте запуска):

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
# для runtime Torch лучше не держать CUDA lib64 в LD_LIBRARY_PATH постоянно

hash -r
nvcc -V   # должен показать release 12.8
```


## 2. Conda + Python 3.12

```bash
# если miniconda ещё нет:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n 1cat-vllm python=3.12
conda activate 1cat-vllm
python -m pip install -U pip setuptools wheel
```


## 3. Установка wheel 1Cat-vLLM

```bash
mkdir -p ~/downloads/1cat && cd ~/downloads/1cat

# скачайте актуальный wheel с релизов:
# https://github.com/1CatAI/1Cat-vLLM/releases/latest
# пример имени: 1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl

wget -O 1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl \
  "https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.3.0/1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl"

python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./1cat_vllm-1.3.0-cp312-cp312-linux_x86_64.whl
```

Wheel уже тянет Torch cu128 и включает `flash_attn_v100` + SM70-ядра.

Проверяем:

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

> Проверка (из `~` или `/tmp`, **не** из клона репозитория)



## (Опционально) 4. Диск под модели (`/mnt/storage`)

Если модели хранятся на отдельном SSD, смонтированном в `**/mnt/storage**`.  
Путь к весам: `**/mnt/storage/models/...**`.

### 4.1. Автомонт при загрузке

```bash
# UUID вашего sda1 (пример):
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

Опционально — симлинк, чтобы старые пути `$HOME/models` тоже работали:

```bash
ln -sfn /mnt/storage/models ~/models
```

Опционально — кеш Hugging Face на тот же диск:

```bash
mkdir -p /mnt/storage/hf-cache
# один раз, если уже есть кеш на системном NVMe:
# rsync -aH ~/.cache/huggingface/ /mnt/storage/hf-cache/
# mv ~/.cache/huggingface ~/.cache/huggingface.bak
ln -sfn /mnt/storage/hf-cache ~/.cache/huggingface
```

### 4.2 Перенос с системного диска 

Перенос уже скачанных моделей с системного диска:

```bash
rsync -aH --info=progress2 ~/models/ /mnt/storage/models/
# затем: mv ~/models ~/models.bak && ln -sfn /mnt/storage/models ~/models
```


## 5. Пример скачивания моделей 

```bash
# === 27B FP16 (макс. качество dense; нужно ≥60 GiB свободно) ===
hf download Qwen/Qwen3.8-27B \
  --local-dir /mnt/storage/models/Qwen3.8-27B
```

> при необходимости: `hf auth login` -> может потребоваться для скачивания Medgemma-27b


После скачивания обязательно проверьте наличие `config.json` и shard-файлов:


```bash
test -f /mnt/storage/models/Qwen3.8-27B/config.json && echo "27B-FP16 OK" || echo "27B-FP16 BROKEN"

du -sh /mnt/storage/models/Qwen3.8-27B
```

Если каталог пустой или без `config.json`, vLLM выдаст ошибку вида:

`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/mnt/storage/models/...'


## 6. Скрипты запуска (4×V100)

```bash
mkdir -p ~/bin ~/logs
```


### 6.1. Общий env — `~/bin/1cat-env.sh`

```bash
#!/usr/bin/env bash
# source ~/bin/1cat-env.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# важно: порядок устройств = PCI bus
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# не смешивать системный cuBLAS с pip-сборкой Torch
unset LD_LIBRARY_PATH

# первый JIT FlashQLA (если понадобится) — компилятор ≤14
# на 24.04 обычно gcc-13 — ок; при проблемах:
# export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13
# export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"

cd ~   # НЕ запускать из checkout исходников 1Cat-vLLM
```


### 6.2. Скрипт для запуска Qwen3.6-27B FP16 — `~/bin/serve-qwen27b-fp16.sh`

Официальные веса [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B), без квантизации.  

Модель **multimodal** (text + vision).

**Скрипт** `~/bin/serve-qwen27b-fp16.sh` (по умолчанию text-only — стабильнее на V100):

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/bin/1cat-env.sh

MODEL="${MODEL:-/mnt/storage/models/Qwen3.8-27B}"
PORT="${PORT:-8000}"
# ENABLE_VISION=1 — анализ картинок (см. ниже)
ENABLE_VISION="${ENABLE_VISION:-0}"
# ENABLE_YARN=1 — контекст >256K (factor 2->524K / 4->1M); см. блок YaRN
ENABLE_YARN="${ENABLE_YARN:-0}"
YARN_FACTOR="${YARN_FACTOR:-2.0}"   # 2.0 или 4.0

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
  # без YaRN: дефолт 128K; native max 262144
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
  # до N картинок на один prompt; без этого vision-запросы отклоняются/ломаются
  ARGS+=(
    --limit-mm-per-prompt '{"image":2}'
    --skip-mm-profiling
  )
  # vision ест VRAM — при OOM снизьте CONTEXT_LENGTH / max-num-seqs
else
  ARGS+=(--language-model-only)
fi

if [[ "$ENABLE_YARN" == "1" ]]; then
  # официальный рецепт Qwen: YaRN через --hf-overrides (--rope-scaling устарел)
  ARGS+=(
    --hf-overrides "{\"text_config\": {\"rope_parameters\": {\"mrope_interleaved\": true, \"mrope_section\": [11, 11, 10], \"rope_type\": \"yarn\", \"rope_theta\": 10000000, \"partial_rotary_factor\": 0.25, \"factor\": ${YARN_FACTOR}, \"original_max_position_embeddings\": 262144}}}"
  )
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
```

- **`--limit-mm-per-prompt '{"image":5}'`** — лимит multimodal-вложений на один запрос: максимум **5 изображений**. Без флага (и без `--language-model-only`) дефолт vLLM часто жёстче/неудобен; с `image:5` можно слать до пяти картинок в `messages[].content`.
- `--skip-mm-profiling` — не профилировать MM-память на старте (быстрее старт, меньше риск OOM на V100).

Установка прав на запуск:

```bash
chmod +x ~/bin/serve-qwen27b-fp16.sh
```

---

Немного подробностей про YARN

Нативный лимит `Qwen3.8-27B`: **262144**. Дальше — **YaRN** через `--hf-overrides` (флаг `--rope-scaling` в новых vLLM **не работает**).

| `factor` | `--max-model-len` | Окно     |
| --------- | ----------------- | -------- |
| `2.0`     | `524288`          | ~524K    |
| `4.0`     | `1010000`         | ~1M      |

**Важно**

- YaRN **статический**: чуть портит качество на **коротких** запросах. Включайте только когда реально нужен контекст **>256K**.
- Нужен `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, иначе vLLM отклонит `max_model_len` выше derived.
- На **4×V100 32GB** FP16-веса ~54 GiB; KV на 524K/1M часто **не влезет**. Реалистично: `factor: 2.0` + урезанный `CONTEXT_LENGTH`, либо `--kv-cache-dtype fp8_e5m2`. На 1M почти наверняка OOM без квант-KV / меньшей длины.

```bash
  --kv-cache-dtype fp8_e5m2
```

Не включайте `--calculate-kv-scales` без явной нужды. На MiniMax можно попробовать, если не хватает KV под контекст.

---

## 7. Запуск

```bash

# text-only (дефолт, native ≤256K):
~/bin/serve-qwen27b-fp16.sh

# vision (картинки):
ENABLE_VISION=1 CONTEXT_LENGTH=65536 ~/bin/serve-qwen27b-fp16.sh

# фон (vision):
# ENABLE_VISION=1 nohup ~/bin/serve-qwen27b-fp16.sh > ~/logs/1cat-27b-fp16-vl.log 2>&1 &

# native 256K text-only:
# CONTEXT_LENGTH=262144 ~/bin/serve-qwen27b-fp16.sh
# при OOM: CONTEXT_LENGTH=65536 ~/bin/serve-qwen27b-fp16.sh
```


Проверка API (имя = `--served-model-name`):

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Ответь одним словом: столица Франции?"}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Пример с картинкой (сервер должен быть запущен с `ENABLE_VISION=1`):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Опиши изображение кратко."},
        {"type": "image_url", "image_url": {"url": "https://www.example.com/demo.jpg"}}
      ]
    }],
    "max_completion_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

С картинкой с диска 

```bash
IMG=./pic5.jpg
B64=$(base64 -i "$IMG" | tr -d '\n')

# проверка, что строка не пустая:
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
          {type: "text", text: "Опиши изображение кратко."},
          {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $b64)}}
        ]
      }],
      max_completion_tokens: 128,
      chat_template_kwargs: {enable_thinking: false}
    }')"
```


## (Опционально) 8. Автозапуск - systemd 

> Впишите своего пользователя вместо **YOUR_USER**

Файл `/etc/systemd/system/1cat-vllm.service`:

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

## Типичные ошибки


### `HFValidationError: Repo id must be in the form...`

Каталог модели отсутствует, пустой или без `config.json`. Перескачайте модель и проверьте файлы.

### `CUBLAS_STATUS_INVALID_VALUE`

Смешение системных CUDA-библиотек и pip Torch. Сделайте `unset LD_LIBRARY_PATH` перед запуском. Не ставьте в `LD_LIBRARY_PATH` `/usr/local/cuda-12.8/lib64` для runtime serve.

### Import из исходников вместо wheel

Запуск из директории с клоном `1Cat-vLLM`. Перейдите в `~` или `/tmp`.

### GPU занята / Free memory too low

Остановите другие процессы (vLLM-зомби, ComfyUI на тех же картах). `nvidia-smi` → `kill -9` лишних PID.