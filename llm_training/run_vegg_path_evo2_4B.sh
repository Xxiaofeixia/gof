#!/bin/bash
#SBATCH -J BioReason_S1
#SBATCH -p GPU3
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -o job_s1.%j.out
#SBATCH -e job_s1.%j.err

source /gpfs/hpc/home/lijc/mapengtao/miniconda3/etc/profile.d/conda.sh
conda activate bioreason_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE="/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface/datasets"
export HF_HOME="/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TEXT_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/Qwen3-4B"
DNA_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/evo2_1b_base"

echo "========== 训练前 GPU 状态 =========="
nvidia-smi
echo "====================================="

# python train_dna_qwen_vegg.py \
#     --stage 1 \
#     --text_model_name $TEXT_MODEL \
#     --dna_model_name $DNA_MODEL \
#     --dna_is_evo2 True \
#     --dataset_type variant_effect_coding \
#     --wandb_project nt-qwen-gof-lof-binary \
#     --max_epochs 3 \
#     --batch_size 1 \
#     --num_gpus 1 \
#     --model_type dna-llm \
#     --max_length_dna 512 \
#     --max_length_text 2048\
#     --truncate_dna_per_side 256 \
#     --return_answer_in_batch True \
#     --num_workers 4 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 2e-5 \
#     --lora_rank 64 \
#     --lora_alpha 128 \
#     --strategy deepspeed_stage_2

# 阶段一提交完成后，自动提交阶段二
sbatch --job-name=BioReason_S2 \
    --partition=GPU3 \
    --gres=gpu:1 \
    --qos=normal \
    --output=job_s2.%j.out \
    --error=job_s2.%j.err \
    --wrap="
source /gpfs/hpc/home/lijc/mapengtao/miniconda3/etc/profile.d/conda.sh
conda activate bioreason_env
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\$LD_LIBRARY_PATH
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE='/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface/datasets'
export HF_HOME='/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface'
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python train_dna_qwen_vegg.py \
    --stage 2 \
    --text_model_name $TEXT_MODEL \
    --dna_model_name $DNA_MODEL \
    --dna_is_evo2 True \
    --dataset_type variant_effect_coding \
    --wandb_project nt-qwen-gof-lof-binary \
    --max_epochs 3 \
    --batch_size 1 \
    --num_gpus 1 \
    --model_type dna-llm \
    --max_length_dna 512 \
    --max_length_text 2048 \
    --truncate_dna_per_side 256 \
    --return_answer_in_batch True \
    --num_workers 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --strategy deepspeed_stage_2
"
