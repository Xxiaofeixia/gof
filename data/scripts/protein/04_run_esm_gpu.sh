#!/bin/bash
#SBATCH --job-name=ESM_DDG_P04
#SBATCH --partition=GPU3
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/04_esm_ddg_%j.log

set -eo pipefail

# 第 04 步需要 GPU：读取 03_BIOREASON_with_RSA.csv，
# 用 ESM-2 计算错义突变的稳定性变化分数 ESM_DDG_Score。
PROJECT_DIR="/gpfs/hpc/home/lijc/mapengtao/gof"
SCRIPT_DIR="${PROJECT_DIR}/data/scripts/protein"

source ~/.bashrc
conda activate ddg_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${SCRIPT_DIR}"
python 04_calc_esm_score.py
