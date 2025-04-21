#!/usr/bin/env sh

train_data_path='./configs/data.yaml'

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
# check_path=/your/path/to/checkpoints
batch_size=16
snr_type=lognorm
lr=2e-4
precision=bf16
size=1024

exp_name=${model}_bs${batch_size}_lr${lr}_${precision}
mkdir -p results/"$exp_name"

NNODES=1
NPROC_PER_NODE=4
MASTER_PORT=1234 #1234
NODE_RANK=0

# python -u finetune.py \
#     --master_port 18182 \
#     --global_bsz_${size} 128 \
#     --micro_bsz_${size} 16 \
#     --model ${model} \
#     --lr ${lr} --grad_clip 2.0 \
#     --data_path ${train_data_path} \
#     --results_dir results/"$exp_name" \
#     --data_parallel sdp \
#     --max_steps 3000000 \
#     --ckpt_every 1000 --log_every 10 \
#     --precision ${precision} --grad_precision fp32 --qk_norm \
#     --global_seed 20241207 \
#     --num_workers 12 \
#     --cache_data_on_disk \
#     --snr_type ${snr_type} \
#     --checkpointing \
#     2>&1 | tee -a results/"$exp_name"/output.log
#     # --init_from ${check_path} \

torchrun \
    --standalone \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    finetune.py \
    --master_port $MASTER_PORT \
    --global_bsz_${size} 128 \
    --micro_bsz_${size} 16 \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 \
    --data_path ${train_data_path} \
    --results_dir results/"$exp_name" \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 1000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20241207 \
    --num_workers 12 \
    --cache_data_on_disk \