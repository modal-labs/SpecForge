SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

MODEL_PATH=/target/glm46_language_cascade_rl_iter23_hf
MODEL_NAME=glm46_language_cascade_rl_iter23_hf
DATASET_PATH=/glm4_5/dataset/regenerated/glm_dataset_regenerated_15k.jsonl
OUTPUT_ROOT_DIR=/specforge/$MODEL_NAME
MAX_LENGTH=32768
TP_SIZE=$((8 * $NUM_NODES))

torchrun \
    --nproc_per_node 8 \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/glm4_5-355b-a32b-eagle3.json \
    --train-data-path $DATASET_PATH \
    --train-hidden-states-path $OUTPUT_ROOT_DIR/cache/hidden_states \
    --output-dir $OUTPUT_ROOT_DIR/outputs \
    --num-epochs 10 \
    --draft-micro-batch-size 1 \
    --draft-global-batch-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length $MAX_LENGTH \
    --chat-template glm4_moe \
    --resume
