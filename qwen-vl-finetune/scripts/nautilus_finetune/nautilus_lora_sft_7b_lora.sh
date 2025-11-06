#!/bin/bash
# Work_dirs
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=4

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration Using HuggingFace model ID / Local model path
pretrain_llm=Qwen/Qwen2.5-VL-7B-Instruct
dino_vitl_weight="weight/dino_vitl.pth"

# Training hyperparameters
lr=2e-5
# Nautilus module lr
nautilus_lr=2e-7 
batch_size=4
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets="nautilus_instruct%100"

# Output configuration
run_name="nautilus_qwen2_5vl_sft"
output_dir=./output/nautilus_qwen2_5vl_sft

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${pretrain_llm}" \
    --dino_path "${dino_vitl_weight}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 1048992 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --nautilus_lr ${nautilus_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard \
    --use_lora True \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05"
# --resume_from_checkpoint checkpoint_path 

# Launch training
torchrun --nproc_per_node=${NNODES} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}