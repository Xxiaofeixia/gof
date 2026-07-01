#!/usr/bin/env bash
# 等第 10 步 API 推理链全部完成后，提交第 1 阶段 SFT 训练。
# 第 1 阶段训练脚本会在完成后自动提交第 2 阶段。

set -euo pipefail

PROJECT_DIR="/gpfs/hpc/home/lijc/mapengtao/gof"
DONE_MARKER="${PROJECT_DIR}/data/processed/protein_10_full.done"
STAGE2_LOG="${PROJECT_DIR}/data/processed/protein_10_stage2_full.log"

cd "${PROJECT_DIR}"
mkdir -p logs data/processed

echo "等待第 10 步推理链完成: ${DONE_MARKER}"
while [[ ! -f "${DONE_MARKER}" ]]; do
  if [[ -f "${STAGE2_LOG}" ]]; then
    tail -n 3 "${STAGE2_LOG}" || true
  fi
  sleep 300
done

echo "检测到第 10 步完成，提交第 1 阶段 SFT 训练。"
sbatch --export=ALL,STAGE=1 llm_training/run_sft_reasoning_4B.sh
