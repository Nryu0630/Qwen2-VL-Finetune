#!/bin/bash

# Activate conda environment to ensure all packages are available
source /home/shouac/anaconda3/etc/profile.d/conda.sh
conda activate qwen

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# Set default to 1 if no GPUs detected (for CPU training/testing)
if [ "$NPROC_PER_NODE" -eq 0 ]; then
    NPROC_PER_NODE=1
fi

# DeepSpeed configuration
deepspeed=/home/shouac/dyao/Qwen2.5-VL/qwen-vl-finetune/scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters optimized for small dataset (138 samples)
lr=5e-6                    # Higher learning rate for small dataset
batch_size=1               # Reduced batch size to save memory
grad_accum_steps=8         # Increased gradient accumulation to maintain effective batch size

# Training entry point
entry_file=/home/shouac/dyao/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py

# Dataset configuration
datasets="backdoor%100"  # Use 100% of the backdoor dataset

# Output configuration
run_name="qwen2.5vl-backdoor-attack"
output_dir=./checkpoints_backdoor_attack

# Training arguments optimized for backdoor training
args="--deepspeed ${deepspeed} \
--model_name_or_path ${llm} \
--dataset_use ${datasets} \
--data_flatten false \
--tune_mm_vision false \
--tune_mm_mlp true \
--tune_mm_llm true \
--bf16 true \
--output_dir ${output_dir} \
--num_train_epochs 10 \
--per_device_train_batch_size ${batch_size} \
--per_device_eval_batch_size ${batch_size} \
--gradient_accumulation_steps ${grad_accum_steps} \
--max_pixels 451584 \
--min_pixels 12544 \
--eval_strategy no \
--save_strategy steps \
--save_steps 25 \
--save_total_limit 3 \
--learning_rate ${lr} \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--max_grad_norm 1.0 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--model_max_length 2048 \
--gradient_checkpointing true \
--dataloader_num_workers 2 \
--run_name ${run_name} \
--report_to none \
--torch_compile true \
--torch_compile_mode reduce-overhead \
--optim adamw_torch_fused"

echo "Starting backdoor dataset training..."
echo "=============================================="
echo "Model: ${llm}"
echo "Dataset: ${datasets} (138 samples: 18 backdoor + 120 clean)"
echo "Output directory: ${output_dir}"
echo "Batch size: ${batch_size}"
echo "Learning rate: ${lr}"
echo "Number of GPUs: ${NPROC_PER_NODE}"
echo "Training epochs: 10"
echo "Save steps: 25"
echo "Attention: SDPA (Standard)"
echo "Data flattening: disabled"
echo "=============================================="
echo ""

# Get the full path to Python in the current environment
PYTHON_PATH=$(which python)

# Launch training
if [ "$NPROC_PER_NODE" -eq 1 ]; then
    # Single GPU or CPU training
    ${PYTHON_PATH} ${entry_file} ${args}
else
    # Multi-GPU training - use the full Python path to ensure correct environment
    ${PYTHON_PATH} -m torch.distributed.run \
             --nproc_per_node=${NPROC_PER_NODE} \
             --master_addr=${MASTER_ADDR} \
             --master_port=${MASTER_PORT} \
             ${entry_file} ${args}
fi

echo ""
echo "=============================================="
echo "Backdoor training completed!"
echo "Model saved to: ${output_dir}"
echo "Dataset used: conversation dataset"
echo "Total samples trained: 138 (18 backdoor + 120 clean)"
echo "==============================================" 