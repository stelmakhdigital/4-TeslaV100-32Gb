# Установка и запуск 1Cat-vLLM на Ubuntu 24.04

**Целевое железо:** 4× Tesla V100 32GB  
**Версия:** 1Cat-vLLM 1.3.0  
**Официальные референсы:** [README](https://github.com/1CatAI/1Cat-vLLM) · [Releases](https://github.com/1CatAI/1Cat-vLLM/releases)

> Qwen3.8-27B-**NVFP4 + DFlash2** (1Cat **1.5.0** wheel, TP4) — отдельный гайд: [GUIDE-NVFP4-DFLASH2-RU.md](./GUIDE-NVFP4-DFLASH2-RU.md). Wheel 1.3.0 DFlash2 не поднимает.


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

> Этот гайд ставит **1.3.0** (AWQ/MTP). Для Qwen3.8 **NVFP4 + DFlash2** нужен wheel **1.5.0** в отдельном env — [GUIDE-NVFP4-DFLASH2-RU.md](./GUIDE-NVFP4-DFLASH2-RU.md).

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

---

# Часть 2. llama.cpp — альтернатива: Q8_0 GGUF, MTP, 500K контекст

> Часть полностью независима от vLLM-разделов: conda/vLLM не нужны, достаточно CUDA 12.8 (раздел 1) и инструментов сборки.
>
> Сценарий: вытащить максимальный контекст из 4×V100 (Q8-веса + квантованный KV) и/или использовать **MTP** (Multi-Token Prediction) — спекулятивное декодирование через MTP-головку модели.
>
> ⚠ llama.cpp должна поддерживать именно вашу модель (и MTP) — берите свежий релиз, changelog есть на [Releases](https://github.com/ggml-org/llama.cpp/releases).

## 9. Сборка llama.cpp (V100 = SM70)

```bash
sudo apt install -y git cmake

git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
git fetch --tags
git checkout <тег-актуального-релиза>   # фиксируйте релиз, а не плавающий master

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --config Release -j"$(nproc)"
```

- **`-DCMAKE_CUDA_ARCHITECTURES=70`** — компиляция только под V100 (SM70). Без флага CMake автоопределяет GPU через `nvidia-smi` и соберёт то же, но явная фиксация — страховка от «построил не под то железо».
- **Flash attention на V100 не работает** — FA-ядра ggml-cuda нацелены на SM80+. Не форсите `-fa on`: дефолтный путь attention вполне достаточен; в лог сборки может попасть пометка, что FA недоступна для этого arch (это нормально).

Установка бинарников:

```bash
sudo install -D -m755 build/bin/llama-* /usr/local/bin/
llama-server --version    # должно показать cuda: yes
```

## 10. Веса: HF → GGUF Q8_0

### 10.1. Окружение для конвертера

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n llama-gguf python=3.12
conda activate llama-gguf
python -m pip install -U pip
python -m pip install gguf sentencepiece protobuf tokenizers safetensors
```

### 10.2. Конвертация и квантизация

```bash
mkdir -p /mnt/storage/models/gguf
cd ~/llama.cpp   # конвертер использует относительные импорты из репозитория

# 1) HF → f16 GGUF (промежуточный файл, ~55 GB)
python convert_hf_to_gguf.py /mnt/storage/models/Qwen3.8-27B \
  --outtype f16 \
  --outfile /mnt/storage/models/gguf/Qwen3.8-27B-f16.gguf

# 2) f16 → Q8_0 (финальный файл, ~28–29 GB)
# аргументы позиционные: вход выход тип  (флага --output нет)
llama-quantize \
  /mnt/storage/models/gguf/Qwen3.8-27B-f16.gguf \
  /mnt/storage/models/gguf/Qwen3.8-27B-Q8_0.gguf \
  Q8_0

du -sh /mnt/storage/models/gguf/*
```

- **MTP**: если в HF-весах есть MTP-тензоры (`mtp.*`), конвертер сохранит их в GGUF — llama.cpp определяет MTP автоматически при старте (строки про MTP/draft в логе). Если их нет — GGUF собран старым конвертером, пересоберите с новым llama.cpp.
- Альтернатива: не конвертировать, а скачать готовый Q8_0 GGUF (с MTP) из сообщества Hugging Face.

## 11. Скрипт запуска — `~/bin/serve-qwen27b-llamacpp.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

SPLIT_MODE="${SPLIT_MODE:-layer}"            # none|layer|row|tensor
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # ← вот где «кол-во устройств»
MODEL="${MODEL:-/mnt/storage/models/gguf/Qwen3.8-27B-Q8_0.gguf}"
PORT="${PORT:-8001}"
# 500K контекст: YaRN ×2 относительно нативного 262144
CTX="${CTX:-524280}"
NATIVE_CTX="${NATIVE_CTX:-262144}"
# MTP: сколько draft-токенов на шаг (0 = выключено)
DRAFT_MAX="${DRAFT_MAX:-3}"
# квантизация KV-кэша: обязателена при большом контексте
KV_TYPE="${KV_TYPE:-q8_0}"
# скрипт сам задаёт -ngl и -c → автоподбор --fit on конфликтует и abort
FIT="${FIT:-off}"

[[ -f "$MODEL" ]] || { echo "No model file: $MODEL"; exit 1; }

ARGS=(
  --model "$MODEL"
  --alias qwen3.8-27b        # имя модели в API
  --host 0.0.0.0
  --port "$PORT"
  -ngl 99                    # все слои на GPU
  -c "$CTX"                  # размер контекста
  -sm "$SPLIT_MODE"
  --fit "$FIT"
  -np 1                      # 1 слот: при 500K параллелизм не влезает
  -b 2048 -ub 512            # prompt batch / ubatch
  --cache-type-k "$KV_TYPE"  # квант KV (K)
  --cache-type-v "$KV_TYPE"  # квант KV (V)
  --jinja                     # chat-шаблон из GGUF (tool-calls, thinking)
  #--chat-template-kwargs '{"enable_thinking":false}'
  --cont-batching
)

# контекст выше нативного — YaRN (тот же подход, что и у vLLM)
if [[ "$CTX" -gt "$NATIVE_CTX" ]]; then
  ARGS+=(
    --rope-scaling yarn
    --rope-scale 2.0
    --rope-freq-base 10000000
    --yarn-orig-ctx "$NATIVE_CTX"
  )
fi

# MTP: головка в том же GGUF; второй файл не нужен
if [[ "$DRAFT_MAX" -gt 0 ]]; then
  ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$DRAFT_MAX")
fi

exec llama-server "${ARGS[@]}"
```

```bash
chmod +x ~/bin/serve-qwen27b-llamacpp.sh
```

```bash
# целевая конфигурация — все 4×V100, 500K:
./serve-qwen27b-llamacpp.sh

# 2 GPU: 500K не влезет. Сначала меньший контекст:
CUDA_VISIBLE_DEVICES=0,1 CTX=65536 ./serve-qwen27b-llamacpp.sh

# tensor-parallel: нужен Flash Attention (SM80+), на V100 не работает.
# CUDA_VISIBLE_DEVICES=0,1 SPLIT_MODE=tensor KV_TYPE=f16 CTX=65536 ./serve-qwen27b-llamacpp.sh
```

### Про 500K контекст

- Нативное окно `Qwen3.8-27B` = **262144**. Выше — **YaRN** (factor 2 → ~524K), тот же приём, что и в vLLM-части. Флаги YaRN дублируют `rope_parameters` модели (`rope_theta = 1e7`, натив 262144); у другой модели — подставьте её значения.
- **KV-бюджет — главное, что решит, влезет ли это:**

  `KV на токен = 2 × n_layers × n_kv_heads × head_dim × байт(типа KV)`

  Реальный размер `llama-server` печатает в логе при старте (`KV self size (MB)`). При 500K с `q8_0` KV — десятки гигабайт; с `f16`-KV 500K рядом с Q8-весами в 4×32 GB **не влезает**. Если OOM: урезайте `CTX` или ставьте `KV_TYPE=q4_0` (качество заметно страдает).

### Про MTP

- MTP = модель сама предсказывает несколько токенов за один шаг через встроенную MTP-головку. В свежем llama.cpp: `--spec-type draft-mtp --spec-draft-n-max N` (второй GGUF не нужен). Старые `--draft` / `--draft-max` **удалены**.
- `N` — сколько токенов черновиком (0–4). На V100 **3** обычно оптимум: смотрите принятие (acceptance) и скорость в логе. Если прирост около нуля — `DRAFT_MAX=1` или `0`.
- MTP ускоряет в основном **генерацию**; prefill от него не зависит.

## 12. Запуск и проверка

```bash
# 500K + MTP (целевая конфигурация):
~/bin/serve-qwen27b-llamacpp.sh

# то же без MTP (для сравнения скорости):
DRAFT_MAX=0 ~/bin/serve-qwen27b-llamacpp.sh

# нативные 256K, без YaRN:
CTX=262144 ~/bin/serve-qwen27b-llamacpp.sh

# фон:
nohup ~/bin/serve-qwen27b-llamacpp.sh > ~/logs/llamacpp-q8-500k.log 2>&1 &
tail -f ~/logs/llamacpp-q8-500k.log
```

Проверка API (имя = `--alias`):

```bash
curl -s http://127.0.0.1:8001/v1/models | jq .

curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Ответь одним словом: столица Франции?"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

> API `llama-server` OpenAI-совместим; `Authorization`-заголовок не обязателен (можно выдать ключ через `--api-key`).

Автозапуск — берите unit из раздела 8, заменив имя сервиса и `ExecStart=/home/YOUR_USER/bin/serve-qwen27b-llamacpp.sh`.

## 13. Оптимизации и частые проблемы (llama.cpp)

| Задача | Как |
| ------ | --- |
| Не влезает (OOM по KV) | `CTX=65536 …`, `KV_TYPE=q4_0`, `-np 1` |
| Параллелизм на малом контексте | отдельный скрипт: `-np 4 -c 32768` |
| MTP не даёт прироста | `DRAFT_MAX=1`; есть ли MTP в логе; пересобрать GGUF |
| Бьются tool-calls | `--jinja` (уже в скрипте) |
| Медленный prefill на 500K | ожидается: prefill линеен по контексту; `-b 2048 -ub 512` |
| MoE-модели (если понадобится) | `--n-cpu-moe N` — вынести эксперты на CPU |
| Не хватает RAM | `--mmap` (дефолт); `--mlock` — зафиксировать веса в RAM |

**Типичные ошибки**

- **`n_gpu_layers already set by user to 99, abort`** — `--fit on` (дефолт llama.cpp) хочет уменьшить `-ngl`, но скрипт его уже зафиксировал. Нужен `--fit off` (скрипт ставит сам). Если после этого CUDA OOM — не хватает VRAM: на **2×V100** уберите `CTX=524280`, ставьте `CTX=65536` (или меньше) либо запускайте на **4 GPU**.
- **`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`** — у `--split-mode tensor` автоподбор памяти не работает. На **V100** `tensor` почти наверняка упадёт дальше: нужен Flash Attention (SM80+) и KV без кванта (`f16`/`bf16`). Для V100 — `SPLIT_MODE=layer`.
- **`invalid argument: --rope-frequency-base`** — в актуальном llama.cpp флаг называется `--rope-freq-base`; оригинал контекста — `--yarn-orig-ctx`, не `--rope-scale-original`.
- **`the argument has been removed` / `invalid argument: --draft`** — MTP теперь `--spec-type draft-mtp --spec-draft-n-max N`.
- **В логе нет упоминаний MTP/draft** — MTP-слои не попали в GGUF. Пересоберите GGUF (раздел 10.2) свежим конвертером.
- **`CUDA error` при старте** — бинарник собран не под SM70. Проверьте `-DCMAKE_CUDA_ARCHITECTURES=70` и пересоберите.
- **OOM на весах** — запустили f16-GGUF (~55 GB) вместо Q8_0.
- **`-fa on` не работает / ошибка** — нормально для V100 (SM70), см. раздел 9.

---

# Часть 3. 3D-пайплайн: ComfyUI (референсные картинки) + Hunyuan3D 2.1 (картинка → GLB)

> Картина сервера целиком (4×V100 32GB, `192.168.1.114`):
>
> | GPU | Сервис | Порт | Зачем |
> | --- | ------ | ---- | ----- |
> | 0–1 | llama.cpp + Qwen3.8-27B (Часть 2) | 8001\* | LLM: текст, длинные контексты, MTP |
> | 2 | ComfyUI: SDXL + FLUX.1-schnell | 8188 | референсные картинки |
> | 3 | Hunyuan3D 2.1 (FastAPI) | 8090 | картинка → текстурированный GLB |
>
> \* `ai-status.sh` (раздел 16) проверяет llama.cpp на `:8080` — поправьте строку в скрипте под свой порт или запустите llama.cpp с `PORT=8080`.
>
> **Пайплайн:** LLM описывает объект → ComfyUI рендерит референсную картинку → Hunyuan3D строит 3D-модель с текстурами → на выходе GLB.
>
> ⚠ В этой раскладке llama.cpp остаётся на **2×V100** (0–1): 500K-контекст на двух картах **не влезает** (Часть 2, раздел 11) — запускайте `CUDA_VISIBLE_DEVICES=0,1 CTX=65536 ~/bin/serve-qwen27b-llamacpp.sh` либо переживите до перезапуска на 4 GPU.

## 14. Что устанавливает скрипт и для чего

| Компонент | Куда ставится | Зачем | Размер |
| --------- | ------------- | ----- | ------ |
| HuggingFace CLI (`hf` / `huggingface-cli`) | `~/bin/hf-cli-venv` (только если нет в PATH) | скачивание моделей с Hugging Face; работает с gated-репо через `HF_TOKEN` | — |
| **SDXL base 1.0** (checkpoint) | `/mnt/storage/models/comfyui/checkpoints/` | базовая модель картинок для ComfyUI — **обязательна** | ~6.9 GB |
| **SDXL VAE** | `/mnt/storage/models/comfyui/vae/` | VAE (декод картинок) для SDXL — **обязателен** | ~335 MB |
| **FLUX.1-schnell** (checkpoint) | `…/comfyui/checkpoints/` | более качественные картинки (дистиллированная, быстрая) — опционально, gated | ~23.8 GB |
| **FLUX VAE** (`ae.safetensors`) | `…/comfyui/vae/` | VAE для FLUX — опционально | ~335 MB |
| **T5-XXL (fp16)** | `…/comfyui/clip/t5xxl_fp16.safetensors` | основной текстовый энкодер FLUX. **Строго fp16**: у V100 (SM70) нет fp8 — опционально | ~9.9 GB |
| **CLIP-L** | `…/comfyui/clip/clip_l.safetensors` | дополнительный текстовый энкодер FLUX — опционально | ~250 MB |
| **ComfyUI** (приложение) | `~/apps/ComfyUI` — **клонировать заранее**, скрипт только проверяет наличие | nodal-генерация картинок: Web UI + API на `:8188` | — |
| `comfy-models.json` | `~/apps/ComfyUI/comfy-models.json` | указывает ComfyUI на доп. корень моделей `/mnt/storage/models/comfyui` (структура `checkpoints/ vae/ clip/` совпадает, модели не перемещаются) | — |
| **Hunyuan3D-2.1** (приложение) | `~/apps/Hunyuan3D-2.1` (git clone) | картинка → текстурированная 3D-модель (shape + paint), FastAPI на `:8090` | ~1 GB |
| Веса Hunyuan3D | `/mnt/storage/models/Hunyuan3D-2.1` + симлинк `~/apps/Hunyuan3D-2.1/weights` | веса shape- и paint-моделей | ~20 GB |
| **torch 2.5.1 (cu124)** + `requirements.txt` | `~/apps/Hunyuan3D-2.1/venv` (Python 3.10/3.11) | официально проверенная конфигурация Hunyuan3D. Не 3.12: пины (numpy 1.24.4, bpy 4.0, pymeshlab 2022.2) не собираются | ~2.5 GB |
| `custom_rasterizer` | тот же venv (C++/CUDA-расширение) | дифференцируемый растеризатор для раскраски текстур | — |
| `mesh_inpaint_processor` | `~/apps/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/` | инпейнтинг текстур на меш (pybind11). Компилируется **python'ом из venv**: их `compile_mesh_painter.sh` дёргает системный `python3-config`, из-за чего `.so` получает suffix другого интерпретатора | — |
| **RealESRGAN_x4plus** | `~/apps/Hunyuan3D-2.1/hy3dpaint/ckpt/` | апскейл сгенерированных текстур ×4 | ~64 MB |
| `~/bin/comfyui-gpu2.sh` | `~/bin/` | запуск ComfyUI на GPU 2, порт 8188 | — |
| `~/bin/hunyuan3d-gpu3.sh` | `~/bin/` | запуск Hunyuan3D на GPU 3, порт 8090 | — |
| `~/bin/ai-status.sh` | `~/bin/` | `nvidia-smi` + живость `:8188` / `:8090` / `:8080` (llama.cpp) | — |
| systemd-юниты (опционально) | `~/bin/comfyui-gpu2.service`, `~/bin/hunyuan3d-gpu3.service` | автозапуск после перезагрузки | — |

**Суммарно на диске:** обязательно ~28 GB (SDXL + VAE + веса Hunyuan3D), с FLUX-стеком ~62 GB. Скрипт проверяет, что на `/mnt/storage` свободно ~60 GB.

**Лицензии:** SDXL — Stability Community License (коммерческое использование ок); FLUX.1-schnell — Apache 2.0, **но репо на HF гейтованное**: нужно принять лицензию на странице репозитория + `HF_TOKEN`.

## 15. Предусловия (до запуска скрипта)

1. **ComfyUI уже склонирован** — скрипт его не клонирует, без каталога он `die`-нет:

   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI ~/apps/ComfyUI
   ```

2. **`~/bin/comfy-env.sh`** — окружение ComfyUI (скрипт запуска его сурсит). Пример — отдельный conda-окружение с torch:

   ```bash
   cat > ~/bin/comfy-env.sh <<'EOF'
   #!/usr/bin/env bash
   # env для ComfyUI — поправьте под своё окружение
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate comfyui
   cd ~/apps/ComfyUI
   EOF
   ```

3. **Компилятор** для CUDA-расширений Hunyuan3D:

   ```bash
   sudo apt install -y build-essential
   ```

4. **Python 3.10/3.11 — опционально**: скрипт ищет `python3.10`/`python3.11` сам; если нет, ставит [uv](https://astral.sh/uv) (без root) и тянет CPython 3.10.
5. **~60 GB свободно** на `/mnt/storage` (раздел 4).

## 16. Скрипт `setup-3d-pipeline.sh`

Сохраните как `setup-3d-pipeline.sh` (например, в `~/`) и запустите:

```bash
bash setup-3d-pipeline.sh
```

Скрипт **идемпотентен**: повторный запуск пропускает готовые шаги (скачки/установка) и докачивает только недостающее.

```bash
#!/usr/bin/env bash
#
# setup-3d-pipeline.sh — сервер arkalaust-AI (4x NVIDIA V100 32GB, 192.168.1.114)
#
# Раскладка GPU:
#   GPU 0-1   llama.cpp + Qwen3.8-27B            (для работы с агентом)
#   GPU 2     ComfyUI: SDXL + FLUX.1-schnell     (референсные картинки)
#   GPU 3     Hunyuan3D 2.1 FastAPI              (картинка -> текстурированный GLB)
#
# Запуск:  bash setup-3d-pipeline.sh
# Идемпотентно: повторный запуск пропускает готовые шаги (скачки/установка).
#
# Всё скачивается в /mnt/storage/models, сервисы ставятся в ~/apps.
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

# HuggingFace CLI: предпочитаем новый `hf` (huggingface-cli deprecated),
# фолбэк на старый; если нет ни того ни другого — ставим сами.
HFCLI_VENV="$BIN_DIR/hf-cli-venv"
if command -v hf >/dev/null 2>&1; then
  HFCLI=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HFCLI=(huggingface-cli)
else
  echo "  HuggingFace CLI не найден — устанавливаю"
  if python3 -m pip install --user -q "huggingface_hub[cli]" 2>/dev/null \
     || pip3 install --user -q "huggingface_hub[cli]" 2>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    # PEP 668 (Ubuntu 24.04+): системный python не даёт pip --user —
    # ставим в отдельный venv, систему не трогаем.
    echo "  pip --user недоступен (PEP 668) — создаю venv $HFCLI_VENV"
    python3 -m venv "$HFCLI_VENV" \
      || die "не удалось создать venv (нужен пакет python3-venv: sudo apt install python3.12-venv)"
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
    die "не удалось установить HuggingFace CLI"
  fi
fi
command -v "${HFCLI[0]}" >/dev/null 2>&1 || die "HuggingFace CLI не найден в PATH: ${HFCLI[0]}"

# dl_file <repo> <file> <dest_dir> [required=1|0]
# Скачивание через huggingface-cli (работает и с gated-репо при HF_TOKEN).
dl_file() {
  local repo="$1" file="$2" dest_dir="$3" required="${4:-1}"
  local dest="$dest_dir/$file"
  if [[ -s "$dest" ]]; then
    skip "уже есть: ${dest#"$MODELS_ROOT"/}"
    return 0
  fi
  mkdir -p "$dest_dir"
  echo "  скачиваю: $repo / $file"
  if HF_TOKEN="${HF_TOKEN:-}" "${HFCLI[@]}" download "$repo" "$file" --local-dir "$dest_dir"; then
    ok "сохранено: ${dest#"$MODELS_ROOT"/} ($(du -h "$dest" | cut -f1))"
  elif [[ "$required" == "1" ]]; then
    die "ошибка: $repo / $file (для gated-репо нужен HF_TOKEN: huggingface.co/settings/tokens)"
  else
    echo -e "  \033[1;33m[warn]\033[0m пропущено (опционально, нужен HF_TOKEN): $file"
  fi
}

# ------------------------------------------------------------------
log "0/5 Предпроверка"
command -v git     >/dev/null || die "не найден git"
command -v curl    >/dev/null || die "не найден curl"
command -v python3 >/dev/null || die "не найден python3"
PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "  python3 (системный): $PYVER — для Hunyuan3D в шаге 3 автоматически подберётся 3.10/3.11"
command -v nvidia-smi >/dev/null || die "nvidia-smi не найден"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "  свободно на $MODELS_ROOT: $(df -BG --output=avail "$MODELS_ROOT" | tail -1 | tr -d ' ' ) GB (нужно ~60)"

# ------------------------------------------------------------------
log "1/5 Модели картинок для ComfyUI -> $COMFY_MODELS"
# Лицензии: SDXL — Stability Community (комм. ок), FLUX.1-schnell — Apache 2.0
# (но репо на HF гейтованное: нужен принятый лицензионный чекбокс + HF_TOKEN).
# t5xxl строго fp16 — V100 не поддерживает fp8!
# SDXL — обязательная часть; FLUX-стек опциональный (required=0):
# без токена пропустится с предупреждением, пайплайн от этого не ломается.
dl_file "stabilityai/stable-diffusion-xl-base-1.0" "sd_xl_base_1.0.safetensors" "$COMFY_MODELS/checkpoints"
dl_file "stabilityai/sdxl-vae"                     "sdxl_vae.safetensors"       "$COMFY_MODELS/vae"
dl_file "black-forest-labs/FLUX.1-schnell"         "flux1-schnell.safetensors"  "$COMFY_MODELS/checkpoints" 0
dl_file "black-forest-labs/FLUX.1-schnell"         "ae.safetensors"             "$COMFY_MODELS/vae" 0
dl_file "comfyanonymous/flux_text_encoders"        "t5xxl_fp16.safetensors"     "$COMFY_MODELS/clip" 0
dl_file "comfyanonymous/flux_text_encoders"        "clip_l.safetensors"         "$COMFY_MODELS/clip" 0
ok "модели картинок готовы (SDXL обязательно; FLUX — опционально)"

# ------------------------------------------------------------------
log "2/5 ComfyUI: пути к моделям + скрипт запуска на GPU 2"
[[ -d "$COMFY_DIR" ]] || die "ComfyUI не найден: $COMFY_DIR (переопредели COMFY_DIR=...)"

# ComfyUI подхватит $COMFY_MODELS как дополнительный корень моделей
# (структура checkpoints/ vae/ clip/ совпадает) — ничего не перемещаем.
cat > "$COMFY_DIR/comfy-models.json" <<EOF
{ "base_path": "$COMFY_MODELS" }
EOF
ok "доп. пути моделей: $COMFY_DIR/comfy-models.json"

cat > "$BIN_DIR/comfyui-gpu2.sh" <<'EOF'
#!/usr/bin/env bash
# ComfyUI на GPU 2 — генерация картинок, API на :8188
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
log "3/5 Hunyuan3D 2.1: репозиторий + venv + веса"
if [[ ! -d "$HY3D_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HY3D_DIR"
  ok "склонировано: $HY3D_DIR"
else
  skip "репозиторий есть: $HY3D_DIR"
fi

# --- Python: репо проверено на 3.10; пины (numpy 1.24.4, bpy 4.0, pymeshlab 2022.2)
#     не собираются на 3.12. Ищем 3.10/3.11, иначе тянем CPython 3.10 через uv (без root).
PYBIN=""
for c in python3.10 python3.11; do
  if command -v "$c" >/dev/null 2>&1; then PYBIN="$c"; break; fi
done
USE_UV=""
if [[ -z "$PYBIN" ]]; then
  echo "  python3.10/3.11 в PATH нет — ставлю uv, чтобы достать CPython 3.10 (без root)"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || die "не удалось установить uv (https://astral.sh/uv)"
  USE_UV=1
fi

# venv с другим python (например, 3.12 от предыдущего запуска) — пересоздаём
if [[ -x "$HY3D_DIR/venv/bin/python" ]]; then
  VENV_PYVER="$("$HY3D_DIR/venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_PYVER" != "3.10" && "$VENV_PYVER" != "3.11" ]]; then
    echo "  текущий venv на python $VENV_PYVER (нужен 3.10/3.11) — пересоздаю"
    rm -rf "$HY3D_DIR/venv"
  fi
fi

if [[ ! -d "$HY3D_DIR/venv" ]]; then
  if [[ -n "$USE_UV" ]]; then
    uv venv --python 3.10 "$HY3D_DIR/venv"
  else
    "$PYBIN" -m venv "$HY3D_DIR/venv"
  fi
  ok "venv создан (python: $("$HY3D_DIR/venv/bin/python" --version 2>&1))"
else
  skip "venv есть"
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
  command -v g++ >/dev/null || die "не найден g++ (нужен для CUDA-расширений): sudo apt install build-essential"
  # 1) torch cu124 — официально проверенная конфигурация
  vpip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  # 2) остальное из requirements
  vpip install -r "$HY3D_DIR/requirements.txt"
  # 3) кастомное C++/CUDA-расширение paint: custom_rasterizer
  ( cd "$HY3D_DIR/hy3dpaint/custom_rasterizer" && vpip install -e . )
  # 4) mesh painter. Их compile_mesh_painter.sh вызывает системный python3-config,
  #    из-за чего .so получает suffix другого python — выполняем ту же команду,
  #    но с python из venv (заголовки + suffix от одного и того же интерпретатора).
  ( cd "$HY3D_DIR/hy3dpaint/DifferentiableRenderer" \
    && c++ -O3 -Wall -shared -std=c++11 -fPIC $($VPY -m pybind11 --includes) \
         mesh_inpaint_processor.cpp \
         -o "mesh_inpaint_processor$($VPY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" )
  # 5) чекпоинт RealESRGAN (апскейл текстур)
  mkdir -p "$HY3D_DIR/hy3dpaint/ckpt"
  if [[ ! -s "$HY3D_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]]; then
    curl -L -C - -o "$HY3D_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
  fi
  touch "$HY3D_DIR/venv/.deps-installed"
  ok "зависимости и расширения установлены"
else
  skip "зависимости уже установлены (пересобрать: rm $HY3D_DIR/venv/.deps-installed)"
fi

mkdir -p "$HY3D_WEIGHTS"
if [[ -z "$(ls -A "$HY3D_WEIGHTS" 2>/dev/null)" ]]; then
  # requirements фиксирует huggingface-hub==0.30.2 — в venv уже есть свой huggingface-cli
  "$HY3D_DIR/venv/bin/huggingface-cli" download tencent/Hunyuan3D-2.1 --local-dir "$HY3D_WEIGHTS" \
    || die "не скачались веса Hunyuan3D (если репо gated — нужен токен: hf auth login)"
  ok "веса скачаны (~20 GB) -> $HY3D_WEIGHTS"
else
  skip "веса уже есть: $HY3D_WEIGHTS"
fi

# репозиторий ищет веса в ./weights
ln -sfn "$HY3D_WEIGHTS" "$HY3D_DIR/weights"
ok "символическая ссылка: $HY3D_DIR/weights -> $HY3D_WEIGHTS"

cat > "$BIN_DIR/hunyuan3d-gpu3.sh" <<'EOF'
#!/usr/bin/env bash
# Hunyuan3D 2.1 FastAPI на GPU 3 — картинка -> текстурированный GLB, API на :8090
cd ~/apps/Hunyuan3D-2.1
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=3
exec python fastapi_server.py --host 0.0.0.0 --port 8090 --enable_tex
EOF
chmod +x "$BIN_DIR/hunyuan3d-gpu3.sh"
ok "$BIN_DIR/hunyuan3d-gpu3.sh"

# ------------------------------------------------------------------
log "4/5 systemd-юниты (автозапуск — опционально)"
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
ok "юниты: $BIN_DIR/comfyui-gpu2.service, $BIN_DIR/hunyuan3d-gpu3.service"
echo "  Включить автозапуск (опционально):"
echo "    sudo cp ~/bin/comfyui-gpu2.service ~/bin/hunyuan3d-gpu3.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable --now comfyui-gpu2 hunyuan3d-gpu3"

# ------------------------------------------------------------------
log "5/5 Скрипт статуса"
cat > "$BIN_DIR/ai-status.sh" <<'EOF'
#!/usr/bin/env bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo
curl -s -m 3 http://127.0.0.1:8188/system_stats >/dev/null && echo "ComfyUI    :8188  OK" || echo "ComfyUI    :8188  down"
curl -s -m 3 http://127.0.0.1:8090/docs           >/dev/null && echo "Hunyuan3D  :8090  OK" || echo "Hunyuan3D  :8090  down"
curl -s -m 3 http://127.0.0.1:8080/health         >/dev/null && echo "llama.cpp  :8080  OK" || echo "llama.cpp  :8080  down (если другой порт — поправь здесь)"
EOF
chmod +x "$BIN_DIR/ai-status.sh"
ok "$BIN_DIR/ai-status.sh"

# ------------------------------------------------------------------
log "Готово"
cat <<EOF

Дальше:
  1) ~/bin/comfyui-gpu2.sh      # или: systemctl start comfyui-gpu2
  2) ~/bin/hunyuan3d-gpu3.sh    # или: systemctl start hunyuan3d-gpu3
  3) ~/bin/ai-status.sh         # проверить GPU и порты

Проверить со стороны Mac (порты должны быть доступны по LAN):
  curl http://192.168.1.114:8188/system_stats
  curl http://192.168.1.114:8090/docs

Если fastapi_server.py Hunyuan3D не знает флаг --enable_tex — посмотри
  python fastapi_server.py --help
и поправь ~/bin/hunyuan3d-gpu3.sh.

FLUX.1-schnell пропущен (gated-репо)? Чтобы докачать позже:
  1) huggingface.co — создать аккаунт, токен (settings/tokens, read-достаточно)
  2) на странице black-forest-labs/FLUX.1-schnell принять лицензию
  3) HF_TOKEN=hf_xxxx bash setup-3d-pipeline.sh   # докачает только FLUX-стек
EOF
```

## 17. Запуск, статус, автозапуск

```bash
~/bin/comfyui-gpu2.sh      # ComfyUI на GPU 2 — http://192.168.1.114:8188 (Web UI + API)
~/bin/hunyuan3d-gpu3.sh    # Hunyuan3D на GPU 3 — http://192.168.1.114:8090 (Swagger: /docs)
~/bin/ai-status.sh         # nvidia-smi + живость :8188 / :8090 / :8080
```

Проверка со стороны Mac (порты должны быть доступны по LAN):

```bash
curl http://192.168.1.114:8188/system_stats
curl http://192.168.1.114:8090/docs
```

Автозапуск после перезагрузки (опционально; скрипт уже сгенерировал юниты):

```bash
sudo cp ~/bin/comfyui-gpu2.service ~/bin/hunyuan3d-gpu3.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now comfyui-gpu2 hunyuan3d-gpu3
```

**Примечания**

- `--enable_tex` у Hunyuan3D включает полный конвейер «форма + текстуры»; если флаг неизвестен вашей версии `fastapi_server.py` — посмотрите `python fastapi_server.py --help` и поправьте `~/bin/hunyuan3d-gpu3.sh` (флаги меняются между релизами).
- `ai-status.sh` проверяет llama.cpp на `:8080`, а Часть 2 по умолчанию поднимает её на `8001` — либо поправьте строку в `ai-status.sh`, либо запускайте llama.cpp с `PORT=8080`.
- FLUX-стек пропущен (нет `HF_TOKEN`)? Докачать: принять лицензию на [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) и `HF_TOKEN=hf_xxxx bash setup-3d-pipeline.sh` — докачается только он.
- В раскладке этой части llama.cpp живёт на **2 GPU** (0–1): 500K-контекст там не влезает, см. предупреждение в шапке Части 3.