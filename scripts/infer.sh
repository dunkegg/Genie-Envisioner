# #!/usr/bin/bash

# script_path=${1}
# echo $script_path

# config_path=${2}
# echo $config_path
# ckp_path=${3}
# output_path=${4}
# domain_name=${5}

# echo "Inference on 1 Nodes, 1 GPUs"
# torchrun --nnodes=1 \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     $script_path \
#     --runner_class_path runner/ge_inferencer.py \
#     --runner_class Inferencer \
#     --config_file $config_path \
#     --mode infer \
#     --checkpoint_path $ckp_path \
#     --output_path $output_path \
#     --n_validation 1 \
#     --n_chunk_action 10 \
#     --domain_name $domain_name

#!/usr/bin/bash

script_path=${1}
config_path=${2}
ckp_path=${3}
output_path=${4}
domain_name=${5}
master_port=${6:-29500}   # 默认29500
n_validation=${7:-50}

echo $script_path
echo $config_path
echo $master_port

echo "Inference on 1 Nodes, 1 GPUs"

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --master_port=${master_port} \
    $script_path \
    --runner_class_path runner/ge_inferencer.py \
    --runner_class Inferencer \
    --config_file $config_path \
    --mode infer \
    --checkpoint_path $ckp_path \
    --output_path $output_path \
    --n_validation $n_validation \
    --n_chunk_action 10 \
    --domain_name $domain_name \
    