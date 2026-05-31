#!/bin/bash
#SBATCH --job-name=ESM_DDG_P04          
#SBATCH --partition=GPU3             # 🚨 请务必检查这是否是你们正确的 GPU 队列名！
#SBATCH --gres=gpu:1                 # 申请 1 张 GPU
#SBATCH --nodes=1                    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                    
#SBATCH --time=12:00:00              
#SBATCH --output=esm_calc_%j.log        

# 激活你装了 ESM 和 PyTorch 的专属环境
source ~/.bashrc
conda activate ddg_env

# 运行 Python 脚本
python 04_calc_esm_score.py