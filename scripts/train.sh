# #!/usr/bin/bash
# export MASTER_PORT=29501

# script_path=${1}
# echo $script_path

# config_path=${2}
# echo $config_path

# if [ -z $WORLD_SIZE ]; then
# NGPU=`nvidia-smi --list-gpus | wc -l`
# echo "Training on 1 Nodes, $NGPU GPUs"
# torchrun --nnodes=1 \
#     --nproc_per_node=$NGPU \
#     --node_rank=0 \
#     $script_path \
#     --config_file $config_path
# else
# echo "Training on $WORLD_SIZE Nodes, 8 GPU per Node"
# NGPU=`nvidia-smi --list-gpus | wc -l`
# torchrun --nnodes=$WORLD_SIZE \
#     --nproc_per_node=$NGPU \
#     --node_rank=$RANK \
#     --master-addr $MASTER_ADDR \
#     --master-port $MASTER_PORT \
#     $script_path \
#     --config_file $config_path
# fi
#!/usr/bin/bash
export MASTER_PORT=${MASTER_PORT:-29500}

script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

if [ -z "$WORLD_SIZE" ]; then
    NGPU=`nvidia-smi --list-gpus | wc -l`
    echo "Training on 1 Nodes, $NGPU GPUs, MASTER_PORT=$MASTER_PORT"

    CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port $MASTER_PORT \
        --nnodes=1 \
        --nproc_per_node=2 \
        --node_rank=0 \
        $script_path \
        --config_file $config_path
else
    echo "Training on $WORLD_SIZE Nodes, 8 GPU per Node"
    NGPU=`nvidia-smi --list-gpus | wc -l`

    CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nnodes=$WORLD_SIZE \
        --nproc_per_node=2 \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $script_path \
        --config_file $config_path
fi