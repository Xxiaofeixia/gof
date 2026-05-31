#!/bin/bash
#SBATCH -J BioReason_protein
#SBATCH -p GPU3
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

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

# 【修改这里】：换成今天刚下好的 4B 文本大模型
TEXT_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/Qwen3-4B"

# 【修改这里】：换成论文里最核心的 Evo2-1B（如果你想测 NT，就换成 NT-v2-500m）
DNA_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/evo2_1b_base"

echo "========== 训练前 GPU 状态 =========="
nvidia-smi
echo "====================================="

python train_dna_qwen_vegg.py \
    --text_model_name $TEXT_MODEL \
    --dna_model_name $DNA_MODEL \
    --dna_is_evo2 True \
    --dataset_type variant_effect_coding \
    --wandb_project nt-qwen-gof-lof-binary \
    --max_epochs 3 \
    --batch_size 2 \
    --num_gpus 1 \
    --model_type dna-llm \
    --max_length_dna 1024 \
    --max_length_text 4096\
    --truncate_dna_per_side 512 \
    --return_answer_in_batch True \
    --num_workers 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --strategy deepspeed_stage_2