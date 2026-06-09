#!/bin/bash
#SBATCH -J BioReason_GRPO
#SBATCH -p GPU3
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -o job_grpo_s1.%j.out
#SBATCH -e job_grpo_s1.%j.err

# ============================================================================
# GRPO 强化学习训练 — Variant Effect 两阶段
# ============================================================================
# 阶段一: Pathogenic vs Benign → GRPO 优化推理能力
# 阶段二: GOF vs LOF           → GRPO 优化推理能力
# ============================================================================

source /gpfs/hpc/home/lijc/mapengtao/miniconda3/etc/profile.d/conda.sh
conda activate bioreason_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export HF_DATASETS_CACHE="/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface/datasets"
export HF_HOME="/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TEXT_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/Qwen3-4B"
DNA_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/evo2_1b_base"

# SFT Checkpoints (你刚训练好的)
SFT_STAGE1="/gpfs/hpc/home/lijc/mapengtao/gof/llm_training/checkpoints/nt-qwen-gof-lof-binary-variant_effect_coding-stage1-Qwen3-4B-20260608-105325/nt-qwen-gof-lof-binary-variant_effect_coding-stage1-Qwen3-4B-epoch=00-val_loss_epoch=nan.ckpt"
SFT_STAGE2="/gpfs/hpc/home/lijc/mapengtao/gof/llm_training/checkpoints/nt-qwen-gof-lof-binary-variant_effect_coding-stage2-Qwen3-4B-20260608-152913/nt-qwen-gof-lof-binary-variant_effect_coding-stage2-Qwen3-4B-epoch=00-val_loss_epoch=nan.ckpt"

OUTPUT_DIR="/gpfs/hpc/home/lijc/mapengtao/gof/llm_training/checkpoints"

echo "========== GRPO 训练前 GPU 状态 =========="
nvidia-smi
echo "=========================================="

# ============================================================================
# 阶段一: Pathogenic vs Benign
# ============================================================================
# 参数说明:
#   --sft_lora_r 64 --sft_lora_alpha 128:  匹配 SFT 的 LoRA 配置 (用于正确加载 checkpoint)
#   --lora_r 16 --lora_alpha 32:           GRPO 阶段的 LoRA 配置 (比 SFT 小, 防止过拟合)
#   --num_generations 8:                    每个 prompt 生成 8 个回复进行组内比较
#   --max_completion_length 256:            回复最大 token 数 (分类任务不用太长)
#   --temperature 0.9:                      采样温度 (较高温度 = 更多探索)
#   --beta 0.04:                            KL 散度惩罚系数 (防止偏离 SFT 太远)
#   --learning_rate 1e-6:                   GRPO 学习率 (远小于 SFT, 防止灾难性遗忘)
#   --vllm_gpu_memory_utilization 0.2:      vLLM 只占 20% 显存 (给训练留空间)
#   --vllm_enable_sleep_mode True:          优化步骤时 vLLM 休眠 (节省显存)
# ============================================================================

echo ""
echo "########################################################################"
echo "  阶段一: Pathogenic vs Benign — GRPO 训练"
echo "########################################################################"
echo ""

python -u train_grpo_vegg.py \
    --stage 1 \
    --text_model_name $TEXT_MODEL \
    --dna_model_name $DNA_MODEL \
    --dna_is_evo2 True \
    --sft_checkpoint "$SFT_STAGE1" \
    --sft_lora_r 64 \
    --sft_lora_alpha 128 \
    --sft_lora_dropout 0.0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --max_length_dna 512 \
    --max_length_text 2048 \
    --truncate_dna_per_side 256 \
    --cache_dir "/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 500 \
    --num_generations 8 \
    --max_completion_length 256 \
    --max_prompt_length 2048 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 20 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --epsilon 0.2 \
    --num_iterations 1 \
    --loss_type dr_grpo \
    --scale_rewards group \
    --reward_funcs xmlcount soft_format strict_format concise correctness \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_enable_sleep_mode True \
    --vllm_max_model_len 3000 \
    --bf16 True \
    --gradient_checkpointing True \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --log_completions True \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --output_dir "$OUTPUT_DIR/grpo_stage1_$(date +%Y%m%d-%H%M%S)" \
    --run_name "grpo-vegg-stage1-4B" \
    --report_to wandb \
    --resume_from_checkpoint True

echo ""
echo "========== 阶段一 GRPO 完成 =========="

# ============================================================================
# 阶段二: GOF vs LOF (自动提交)
# ============================================================================
echo ""
echo "提交阶段二 GRPO 训练..."
sbatch --job-name=BioReason_GRPO_S2 \
    --partition=GPU3 \
    --gres=gpu:1 \
    --qos=normal \
    --output=job_grpo_s2.%j.out \
    --error=job_grpo_s2.%j.err \
    --wrap="
source /gpfs/hpc/home/lijc/mapengtao/miniconda3/etc/profile.d/conda.sh
conda activate bioreason_env
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\$LD_LIBRARY_PATH
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export HF_DATASETS_CACHE='/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface/datasets'
export HF_HOME='/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface'
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo '########################################################################'
echo '  阶段二: GOF vs LOF — GRPO 训练'
echo '########################################################################'

python -u train_grpo_vegg.py \
    --stage 2 \
    --text_model_name $TEXT_MODEL \
    --dna_model_name $DNA_MODEL \
    --dna_is_evo2 True \
    --sft_checkpoint '$SFT_STAGE2' \
    --sft_lora_r 64 \
    --sft_lora_alpha 128 \
    --sft_lora_dropout 0.0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --max_length_dna 512 \
    --max_length_text 2048 \
    --truncate_dna_per_side 256 \
    --cache_dir '/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface' \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 500 \
    --num_generations 8 \
    --max_completion_length 256 \
    --max_prompt_length 2048 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 20 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --epsilon 0.2 \
    --num_iterations 1 \
    --loss_type dr_grpo \
    --scale_rewards group \
    --reward_funcs xmlcount soft_format strict_format concise correctness \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_enable_sleep_mode True \
    --vllm_max_model_len 3000 \
    --dna_embedding_layer 'blocks.20.mlp.l3' \
    --bf16 True \
    --gradient_checkpointing True \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --log_completions True \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --output_dir '$OUTPUT_DIR/grpo_stage2_'\$(date +%Y%m%d-%H%M%S) \
    --run_name 'grpo-vegg-stage2-4B' \
    --report_to wandb \
    --resume_from_checkpoint True
"

echo "========== 全部提交完成 =========="
