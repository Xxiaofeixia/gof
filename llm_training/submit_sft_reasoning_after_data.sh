#!/usr/bin/env bash
# 等第 10 步 API 推理链全部完成后，提交第 1 阶段 SFT 训练。
# 第 1 阶段训练脚本会在完成后自动提交第 2 阶段。

set -euo pipefail

PROJECT_DIR="/gpfs/hpc/home/lijc/mapengtao/gof"
DONE_MARKER="${PROJECT_DIR}/data/processed/protein_10_full.done"
STAGE2_LOG="${PROJECT_DIR}/data/processed/protein_10_stage2_full.log"
STAGE1_DATA="${PROJECT_DIR}/data/processed/10_BioReason_protein_Stage1_Binary_Reasoning.csv"
STAGE2_DATA="${PROJECT_DIR}/data/processed/10_BioReason_protein_Stage2_GOF_LOF_Reasoning.csv"

cd "${PROJECT_DIR}"
mkdir -p logs data/processed

echo "等待第 10 步推理链完成: ${DONE_MARKER}"
while [[ ! -f "${DONE_MARKER}" ]]; do
  if [[ -f "${STAGE2_LOG}" ]]; then
    tail -n 3 "${STAGE2_LOG}" || true
  fi
  sleep 300
done

python - "${STAGE1_DATA}" "${STAGE2_DATA}" <<'PY'
import sys
import pandas as pd

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    if "reasoning_status" not in df.columns or "reasoning_sft" not in df.columns:
        raise SystemExit(f"{path} 缺少 reasoning_status 或 reasoning_sft 列，不提交训练。")
    ok = df["reasoning_status"].eq("ok")
    nonempty = df["reasoning_sft"].fillna("").astype(str).str.strip().ne("")
    bad = int((~ok | ~nonempty).sum())
    if bad:
        raise SystemExit(f"{path} 仍有 {bad} 条未完成推理链，不提交训练；请补跑第 10 步。")
    print(f"{path}: 推理链完整，允许提交训练。")
PY

echo "检测到第 10 步完成，提交第 1 阶段 SFT 训练。"
sbatch --export=ALL,STAGE=1 llm_training/run_sft_reasoning_4B.sh
