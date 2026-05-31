#!/bin/bash
#SBATCH -J BioReason_Task
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

TEXT_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/Qwen3-1.7B"
DNA_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/nt-v2-500m"

echo "========== 训练前 GPU 状态 =========="
nvidia-smi
echo "====================================="

python train_dna_qwen_vegg.py \
    --text_model_name $TEXT_MODEL \
    --dna_model_name $DNA_MODEL \
    --dataset_type variant_effect_coding \
    --ckpt_path /gpfs/hpc/home/lijc/mapengtao/gof/llm_training/checkpoints/nt-500m-qwen3-1.7b-finetune-variant_effect_coding-Qwen3-1.7B-20260411-134740/nt-500m-qwen3-1.7b-finetune-variant_effect_coding-Qwen3-1.7B-epoch=02-val_loss_epoch=0.0210.ckpt \
    --max_epochs 0 \
    --batch_size 1 \
    --num_gpus 1 \
    --model_type dna-llm \
    --max_length_dna 1024 \
    --max_length_text 8192 \
    --truncate_dna_per_side 2000 \
    --return_answer_in_batch True \
    --num_workers 4 \
    --gradient_accumulation_steps 16 \
    --strategy ddp
