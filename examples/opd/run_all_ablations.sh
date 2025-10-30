#!/bin/bash
# Run all OPD ablations in parallel

NUM_GPUS=${1:-8}

if [ $NUM_GPUS -lt 32 ]; then
    echo "Warning: Running all 4 ablations requires 32 GPUs (8 per ablation)"
    echo "With $NUM_GPUS GPUs, running sequentially..."

    ./run_qwen3_baseline.sh $NUM_GPUS
    ./run_qwen3_rkl_only.sh $NUM_GPUS
    ./run_qwen3_mixed.sh $NUM_GPUS
    ./run_qwen3_full_opd.sh $NUM_GPUS
else
    echo "Running all 4 ablations in parallel with $NUM_GPUS GPUs..."
    GPUS_PER_RUN=$((NUM_GPUS / 4))

    CUDA_VISIBLE_DEVICES=0-$((GPUS_PER_RUN-1)) ./run_qwen3_baseline.sh $GPUS_PER_RUN &
    CUDA_VISIBLE_DEVICES=$GPUS_PER_RUN-$((2*GPUS_PER_RUN-1)) ./run_qwen3_rkl_only.sh $GPUS_PER_RUN &
    CUDA_VISIBLE_DEVICES=$((2*GPUS_PER_RUN))-$((3*GPUS_PER_RUN-1)) ./run_qwen3_mixed.sh $GPUS_PER_RUN &
    CUDA_VISIBLE_DEVICES=$((3*GPUS_PER_RUN))-$((4*GPUS_PER_RUN-1)) ./run_qwen3_full_opd.sh $GPUS_PER_RUN &

    wait
fi

echo "All ablations complete!"
