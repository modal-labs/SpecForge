#!/bin/bash
# Ablation 2: Pure OPD (reverse-KL only)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $(dirname $SCRIPT_DIR))
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-1}
TARGET_MODEL=${TARGET_MODEL:-"Qwen/Qwen3-8B"}
TRAIN_DATA=${TRAIN_DATA:-"$ROOT_DIR/cache/dataset/sharegpt.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/outputs/qwen3-8b-rkl-only"}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path $TARGET_MODEL \
    --train-data-path $TRAIN_DATA \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --ttt-length 7 \
    --lambda-ce 0.0 \
    --lambda-rkl 1.0 \
    --beta-hinge 0.0
