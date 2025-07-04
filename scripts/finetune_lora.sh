#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
#MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen2-VL-2B"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export NCCL_P2P_DISABLE=1          # 关闭 GPU‑P2P
export NCCL_IB_DISABLE=1           # 关闭 InfiniBand
export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=7
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /home/yuhong_wang/projects/VLM_Memorization/Qwen2-VL-Finetune/lora_data_origin.json \
    --image_folder /home/yuhong_wang/projects/VLM_Memorization/Qwen2-VL-Finetune/v2/images/origin/train \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /home/yuhong_wang/storage/output/testing_lora \
    --num_train_epochs 20 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --dataloader_num_workers 4 \
    --eval_strategy "epoch" \
    --validation_data_path /home/yuhong_wang/projects/VLM_Memorization/Qwen2-VL-Finetune/lora_data_origin_val.json \
    --validation_image_folder /home/yuhong_wang/projects/VLM_Memorization/Qwen2-VL-Finetune/v2/images/origin/validation \
    --eval_accumulation_steps 4 \
    --eval_steps 0 \
    --per_device_eval_batch_size 4 \
    --do_eval True \
    --predict_with_generate True \
    --generation_max_new_tokens 128 \
    --generation_num_beams 1