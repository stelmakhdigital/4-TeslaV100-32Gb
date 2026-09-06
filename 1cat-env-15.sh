#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-15

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset LD_LIBRARY_PATH
export VLLM_SM70_FLASH_ATTN_V100=1
export VLLM_SM70_NVFP4_TURBOMIND=1

cd ~
