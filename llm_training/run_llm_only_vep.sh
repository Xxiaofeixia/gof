#!/bin/bash
#SBATCH -J LLM_Only
#SBATCH -p GPU3
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

source /gpfs/hpc/home/lijc/mapengtao/miniconda3/etc/profile.d/conda.sh
conda activate bioreason_env

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE="/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface/datasets"
export HF_HOME="/gpfs/hpc/home/lijc/mapengtao/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

TEXT_MODEL="/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models/Qwen3-1.7B"

echo "========== ѵwǰ GPU ״̬ =========="
nvidia-smi
echo "====================================="

python -u train_llm_only.py \
    --text_model_name $TEXT_MODEL \
    --dataset_type variant_effect_coding \
    --max_epochs 3 \
    --batch_size 1 \
    --num_gpus 1 \
    --max_length_text 512 \
    --max_length_dna 512 \
    --truncate_dna_per_side 512 \
    --return_answer_in_batch True \
    --num_workers 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5