#!/usr/bin/env bash
# 使用第 10 步生成的推理链数据训练 BioReason 变异任务。
# STAGE=1: Pathogenic vs Benign；STAGE=2: GOF vs LOF。
# 第 1 阶段训练完成后会自动提交第 2 阶段。

#SBATCH -J gof_sft_reasoning
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 3-00:00:00
#SBATCH -o /gpfs/hpc/home/lijc/mapengtao/gof/logs/sft_reasoning_%x_%j.out
#SBATCH -e /gpfs/hpc/home/lijc/mapengtao/gof/logs/sft_reasoning_%x_%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/hpc/home/lijc/mapengtao/gof"
cd "${PROJECT_DIR}"

mkdir -p logs checkpoints

STAGE="${STAGE:-1}"
TEXT_MODEL_NAME="${TEXT_MODEL_NAME:-Qwen/Qwen3-4B}"
DNA_MODEL_NAME="${DNA_MODEL_NAME:-InstaDeepAI/nucleotide-transformer-v2-500m-multi-species}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WANDB_PROJECT="${WANDB_PROJECT:-gof-bioreason-reasoning-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

STAGE1_DATA="data/processed/10_BioReason_protein_Stage1_Binary_Reasoning.csv"
STAGE2_DATA="data/processed/10_BioReason_protein_Stage2_GOF_LOF_Reasoning.csv"

check_reasoning_data() {
  local file="$1"
  local expected_rows="$2"
  python - "$file" "$expected_rows" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
expected_rows = int(sys.argv[2])
df = pd.read_csv(path)
if len(df) < expected_rows:
    raise SystemExit(f"{path} 行数不足: {len(df)} < {expected_rows}")
if "reasoning_status" not in df.columns:
    raise SystemExit(f"{path} 缺少 reasoning_status 列")
if "reasoning_sft" not in df.columns:
    raise SystemExit(f"{path} 缺少 reasoning_sft 列")
ok = df["reasoning_status"].eq("ok")
nonempty = df["reasoning_sft"].fillna("").astype(str).str.strip().ne("")
bad = int((~ok | ~nonempty).sum())
if bad:
    raise SystemExit(f"{path} 存在 {bad} 条未完成或空推理链样本")
print(f"{path}: {len(df)} 条样本检查通过")
PY
}

echo "检查第 10 步推理链数据..."
check_reasoning_data "${STAGE1_DATA}" 9000
check_reasoning_data "${STAGE2_DATA}" 4500

echo "开始 SFT 训练: stage=${STAGE}, text_model=${TEXT_MODEL_NAME}"

ARGS=(
  --model_type dna-llm
  --dataset_type variant_effect_coding
  --stage "${STAGE}"
  --text_model_name "${TEXT_MODEL_NAME}"
  --dna_model_name "${DNA_MODEL_NAME}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --max_epochs "${MAX_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --num_gpus "${NUM_GPUS}"
  --num_workers "${NUM_WORKERS}"
  --strategy deepspeed_stage_2
  --checkpoint_dir "${PROJECT_DIR}/checkpoints"
  --log_dir "${PROJECT_DIR}/logs"
  --wandb_project "${WANDB_PROJECT}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

python llm_training/train_dna_qwen_vegg.py "${ARGS[@]}"

if [[ "${STAGE}" == "1" ]]; then
  echo "第 1 阶段完成，提交第 2 阶段 SFT 训练。"
  sbatch --export=ALL,STAGE=2 llm_training/run_sft_reasoning_4B.sh
fi
