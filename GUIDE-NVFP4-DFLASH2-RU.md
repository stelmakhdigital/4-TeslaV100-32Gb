# Qwen3.8-27B-NVFP4 + DFlash2 на 4× V100

**Железо:** 4× Tesla V100 32GB (TP=4)  
**Стек:** 1Cat-vLLM **1.5.0** (main после PR [#426](https://github.com/1CatAI/1Cat-vLLM/pull/426) / [#427](https://github.com/1CatAI/1Cat-vLLM/pull/427))  
**Официальный README:** [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)

Это **не** профиль AWQ из `GUIDE-RU.md` (1.3.0). Wheel **1.3.0 DFlash2 не поднимает**: нет `flash_attn_grouped_verify_max_query_tokens`.

На **2 GPU** официальный рецепт не рассчитан.


## Что качается (две модели)

DFlash2 — не отдельный чат-модел. Это **drafter**: target проверяет его черновики.

| Роль | Hugging Face | Локальный путь | Размер | Ревизия |
| ---- | ------------ | -------------- | ------ | ------- |
| **Target** (то, что в `--model`) | [`dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2`](https://huggingface.co/dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2) | `/mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2` | ~24 GiB | `main` |
| **Drafter** (внутри `--speculative-config`) | [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | `/mnt/storage/models/Qwen3.8-27B-DFlash2` | **3.85 GiB** | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |

- Target: тот же mixed NVFP4, что у Unsloth, но **`lm_head` в BF16**. DFlash2 в 1Cat требует неквантованную голову (`CompressedTensorsLinearMethod` на Unsloth — отказ). Не путать с `unsloth/Qwen3.8-27B-NVFP4`.
- Drafter: **BF16**, не квантовать. Зеркало `z-lab/Qwen3.8-27B-DFlash2` то же; 1Cat пинит **incoai** + эту ревизию.

Альтернатива target: [`Inferact/Qwen3.8-27B-NVFP4`](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4) (равномерный NVFP4). Не смешивать с AWQ-INT4 (`cyankiwi/...`).

**Не качать для этого рецепта:** `Qwen/Qwen3.8-27B` FP16, `*-AWQ-INT4`, GPTQ.

На диске свободно **≥40 GiB** (веса + сборка).


## 0. Система

| Компонент | Версия |
| --------- | ------ |
| OS | Ubuntu 24.04 |
| GPU | **4×** V100 32GB |
| CUDA toolkit | **12.8** |
| Python | **3.12** |
| PyTorch | **2.10 + cu128** |
| 1Cat-vLLM | **1.5.0** / `main` с #426+#427 |
| gcc host | **13** (не 15) |

```bash
nvidia-smi
nvcc -V   # release 12.8
```

CUDA 12.8, если ещё нет — как в `GUIDE-RU.md` §1.


## 1. Модели (можно качать параллельно со сборкой)

```bash
python -m pip install -U "huggingface_hub[cli]"
# при 401: hf auth login

hf download dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2 \
  --local-dir /mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2

hf download incoai/Qwen3.8-27B-DFlash2 \
  --revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 \
  --local-dir /mnt/storage/models/Qwen3.8-27B-DFlash2
```

Проверка:

```bash
test -f /mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2/config.json && echo "NVFP4 OK"
test -f /mnt/storage/models/Qwen3.8-27B-DFlash2/config.json && echo "DFlash2 OK"
test -f /mnt/storage/models/Qwen3.8-27B-DFlash2/model.safetensors && echo "DFlash2 weights OK"
du -sh /mnt/storage/models/Qwen3.8-27B-NVFP4-DFlash2 /mnt/storage/models/Qwen3.8-27B-DFlash2
```

Без `config.json` vLLM решит, что путь — HF repo id, и упадёт.


## 2. Runtime: сборка из `main` (wheel 1.5.0 **ещё нет**)

Отдельный conda-env, чтобы не ломать 1.3.0 / AWQ.

На [releases](https://github.com/1CatAI/1Cat-vLLM/releases) последний публичный wheel — **v1.3.0**. Он **не** поднимает DFlash2 (`flash_attn_grouped_verify_max_query_tokens`). Пока нет тега `v1.5.0` — только `git clone` + сборка §2B. Не качайте несуществующий `1cat_vllm-1.5.0-*.whl`.

### 2B. Сборка из `main`

Пакеты хоста:

```bash
sudo apt update
sudo apt install -y git ninja-build cmake patchelf gcc-13 g++-13 build-essential
# patchelf обязателен: без него 1.5.0 отказывается собирать portable wheel
```

Env + Torch **до** компиляции ядер:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n 1cat-vllm-15 python=3.12
conda activate 1cat-vllm-15
python -m pip install -U pip setuptools wheel

python -m pip install --prefer-binary \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
```

В логе pip будет `cuda-bindings==12.9.4` — это **не** toolkit CUDA 12.9. Wheel называется `torch-2.10.0+cu128`, runtime-библиотеки `nvidia-cuda-*-cu12==12.8.*`. Не отменяйте установку. cudnn ~700 MB — качается долго.

Проверка после pip:

```bash
python - <<'PY'
import torch
print(torch.__version__, "cuda", torch.version.cuda)
print("cu128" in torch.__version__, torch.cuda.get_device_capability(0))
PY
# ожидается: 2.10.0+cu128  cuda 12.8  и capability (7, 0)
```

Клон и флаги SM70:

```bash
cd ~
git clone https://github.com/1CatAI/1Cat-vLLM.git
cd ~/1Cat-vLLM
# нужен main с влитыми #426 и #427:
git log --oneline -1
git merge-base --is-ancestor fa344edba8ee460c28d477dcdd2b5d0d90fb4685 HEAD && echo "PR427 OK"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0
export CMAKE_CUDA_ARCHITECTURES=70
export FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS=1
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"

# use_existing_torch.py пачкает requirements/* — перед git pull:
#   git restore pyproject.toml requirements/ && git pull --ff-only
# Повторная сборка без чистки .deps падает: patch already applied.
rm -rf build .deps vllm.egg-info

python use_existing_torch.py
python -m pip install -r requirements/build/cuda.txt
```

Сборка (15–60+ мин). **Не** `pip install vllm` с PyPI.

`use_existing_torch.py` фиксирует torch **только на время compile**. Обычный `pip install -e .` потом резолвит runtime-deps с PyPI и **меняет** `2.10.0+cu128` на `2.13.0+cu130`. CUDA 13 **не** для V100 (нет SM70).

```bash
# FlashAttention-V100 — отдельный пакет, `pip install -e .` его НЕ ставит
python -m pip install --no-build-isolation ./flash-attention-v100 --no-deps

# обязателен --no-deps: иначе pip притащит torch 2.13+cu130
python -m pip install --no-build-isolation -e . --no-deps
python -m pip install -r requirements/common.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
# common.txt тоже может тянуть torch — сразу проверка:
python - <<'PY'
import torch
assert torch.__version__.startswith("2.10.0") and "cu128" in torch.__version__, torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print("OK", torch.__version__, torch.version.cuda)
PY
```

Если assert упал — не запускайте vLLM, верните torch:

```bash
python -m pip install --prefer-binary --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
```

Опционально wheel (как в релизном пайплайне), потом ставить **из `~`**, не из checkout:

```bash
rm -rf build vllm.egg-info dist-cu128-sm70
python -m build --wheel --no-isolation --outdir dist-cu128-sm70
cd ~
python -m pip install --prefer-binary \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ~/1Cat-vLLM/dist-cu128-sm70/1cat_vllm-*.whl
# или vllm-*.whl — смотрите фактическое имя в dist-cu128-sm70
```

Проверка **из `~` или `/tmp`**, не из `~/1Cat-vLLM`:

```bash
cd ~
conda activate 1cat-vllm-15
python - <<'PY'
import sys, torch, vllm
import flash_attn_v100
from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils
from flash_attn_v100 import flash_attn_grouped_verify_max_query_tokens

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "count", torch.cuda.device_count())
print("vLLM:", vllm.__version__)
print("flash_attn_v100:", flash_attn_v100.__version__)
print("DFlash2 grouped verify max Q:", flash_attn_grouped_verify_max_query_tokens())
print("FlashAttention-V100: OK")
PY
```

Ожидание: `DFlash2 grouped verify max Q:` — число (не ImportError). GPU count = **4**.


## 3. Запуск

`~/bin/1cat-env-15.sh`:

```bash
#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-15

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset LD_LIBRARY_PATH
export VLLM_SM70_NVFP4_TURBOMIND=1

cd ~
```

`~/bin/serve-qwen3.8-nvfp4-dflash2.sh` — копия из репо. Скрипт **фиксирует 4 GPU**, ставит `--kv-cache-dtype fp8_e5m2` и один раз патчит guard в `attention.py` (иначе mixed NVFP4 не стартует E5M2, grouped verifier падает в fallback ~30 tok/s).

```bash
chmod +x ~/bin/1cat-env-15.sh ~/bin/serve-qwen3.8-nvfp4-dflash2.sh
~/bin/serve-qwen3.8-nvfp4-dflash2.sh
```

Авто из 1Cat при этом контракте:

```
draft block size = 8
draft width      = 7
selector Top-K   = 16
target KV        = FP8 E5M2 (unit-scale patch в скрипте; без него auto→E4M3 и ~30 tok/s)
draft attention  = FLASH_ATTN_V100
```

Первый старт долгий (CUDA Graph / compile). Не `enforce-eager`.


## 4. Клиент

Имя модели API: **`qwen3.8-27b-dflash2`**.

Рецепт 1Cat / Qwen thinking:

```
temperature=1.0  top_p=0.95  top_k=20
```

Код точнее (опционально): `temperature=0.6`, остальное то же.

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-dflash2",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "ping"}]
  }'
```


## 5. Типичные ошибки

| Симптом | Причина | Что делать |
| ------- | ------- | ---------- |
| `fp8_e5m2 kv-cache is not supported with checkpoint KV scales` | Mixed NVFP4 + guard без патча | Скрипт патчит `attention.py` (unit e5m2). Не откатывайте на `auto` — иначе ~30 tok/s |
| `grouped verifier gate rejected` / `Using the dense LM head` | TP≠4 или KV не E5M2 | 4 GPU, `--tensor-parallel-size 4`, `--kv-cache-dtype fp8_e5m2` |
| `need 4 GPUs for TP4` | `CUDA_VISIBLE_DEVICES` на 2 карты | `0,1,2,3` |
| `DFlash2 requires an unquantized target LM head` | Unsloth квантует `lm_head` | Target: `dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2` (голова BF16) |
| `Numba needs NumPy 2.4 or less. Got NumPy 2.5` | `torch` cu128 притащил numpy 2.5 | `pip install 'numpy>=2.2,<2.5'` |
| `No module named 'flash_attn_v100'` | Собрали только vLLM, не `flash-attention-v100/` | `pip install --no-build-isolation ./flash-attention-v100 --no-deps` |
| `cannot import name 'flash_attn_grouped_verify_max_query_tokens'` | Стоит **1.3.0** (или старый flash_attn_v100) | 1.5.0 / сборка `main` + пакет из `flash-attention-v100/` |
| `Min capability: 75` / SM70 | Не тот квант (AWQ-INT4 compressed-tensors) | NVFP4 unsloth/Inferact, не cyankiwi INT4 |
| Worker не стартует, TP=2 | Нужен **TP=4** | `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| Запуск из `~/1Cat-vLLM` | Подхватывается исходник, не wheel | `cd ~` |
| gcc 15 / host compiler | JIT/NVCC ломается | gcc-13, `CUDAHOSTCXX` |
| OOM | 524K YaRN / vision / util 0.90 | Этот рецепт: 256K, util **0.80**, без YaRN |
| Drafter не грузится | Не та ревизия / квантованный draft | pin `dedf8df…`, BF16 |
| `patchelf not found` | Сборка 1.5.0 | `sudo apt install patchelf` |
| `torch 2.13.0+cu130` / `cuda 13.0` | `pip install -e .` **без** `--no-deps` | `--force-reinstall` torch **2.10.0+cu128** (см. §2B). Не запускать на V100 |


## 6. Чеклист

```
[ ] nvidia-smi: 4× V100
[ ] nvcc 12.8, gcc-13
[ ] conda 1cat-vllm-15, torch 2.10+cu128
[ ] python-check: grouped verify max Q печатается
[ ] …/Qwen3.8-27B-NVFP4-DFlash2/config.json
[ ] …/Qwen3.8-27B-DFlash2/config.json + model.safetensors (~3.85G)
[ ] serve из ~, TP=4, FLASH_ATTN_V100, kv-cache fp8_e5m2, нет grouped verifier rejected
[ ] curl /v1/models → qwen3.8-27b-dflash2
```
