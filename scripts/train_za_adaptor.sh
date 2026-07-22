#!/usr/bin/bash
export MASTER_PORT=${MASTER_PORT:-29501}

script_path=${1:-main.py}
config_path=${2:-configs/ltx_model/za_adaptor_lerobot.yaml}

echo $script_path
echo $config_path

get_nproc_per_node() {
    if [ -n "$NPROC_PER_NODE" ]; then
        echo "$NPROC_PER_NODE"
    elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | sed '/^$/d' | sort -u | wc -l
    else
        nvidia-smi --list-gpus | wc -l
    fi
}

if [ -z "$WORLD_SIZE" ]; then
    NGPU=$(get_nproc_per_node)
    echo "Training Za adaptor on 1 Nodes, $NGPU GPUs, MASTER_PORT=$MASTER_PORT"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}, NPROC_PER_NODE=${NPROC_PER_NODE:-<auto>}"

    torchrun --master_port $MASTER_PORT \
        --nnodes=1 \
        --nproc_per_node=$NGPU \
        --node_rank=0 \
        $script_path \
        --config_file $config_path \
        --runner_class_path runner/za_adaptor_trainer.py \
        --runner_class ZaAdaptorTrainer
else
    NGPU=$(get_nproc_per_node)
    echo "Training Za adaptor on $WORLD_SIZE Nodes, $NGPU GPU processes per Node"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}, NPROC_PER_NODE=${NPROC_PER_NODE:-<auto>}"

    torchrun \
        --nnodes=$WORLD_SIZE \
        --nproc_per_node=$NGPU \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $script_path \
        --config_file $config_path \
        --runner_class_path runner/za_adaptor_trainer.py \
        --runner_class ZaAdaptorTrainer
fi
