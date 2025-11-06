#!/bin/bash

# Bathsize
per_device_train_batch_size=2
num_gpus=4
global_batch_size=128
gradient_accumulation_steps=$((global_batch_size / per_device_train_batch_size / num_gpus))

# Model configuration
pretrain_mm_mlp_adapter=".checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"
llava_weight="liuhaotian/llava-v1.5-7b"
dino_vitl_weight="weight/dino_vitl.pth"

# Training arguments
data_path="Nautilus-instruct-train.json"
image_folder="Nautdata"
nautilus_lr=2e-6

# Use include to select gpus, e.g. --include localhost:0,1,2,3
include="localhost:0,1,2,3"
output_dir="./checkpoints/nautilus_llava_sft"

# Launch training
deepspeed --master_port 29503 --num_gpus ${num_gpus} llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${llava_weight} \
    --version v1 \
    --data_path ${data_path} \
    --image_folder ${image_folder} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${pretrain_mm_mlp_adapter} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --group_by_task False \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --eval_strategy no \
    --nautilus_lr ${nautilus_lr} \
    --dino_vitl_weight ${dino_vitl_weight} \