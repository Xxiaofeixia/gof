#!/usr/bin/env bash
set -eo pipefail

cd /gpfs/hpc/home/lijc/mapengtao/gof
source /gpfs/hpc/home/public/jclabadmin/software/anaconda3/bin/activate gof
source data/scripts/protein/api_key_env.sh

if [[ -z "${DEEPSEEK_API_KEY:-}" || "${DEEPSEEK_API_KEY}" == "PASTE_YOUR_API_KEY_HERE" ]]; then
  echo "请先编辑 data/scripts/protein/api_key_env.sh，填入真实 DEEPSEEK_API_KEY。"
  exit 1
fi

stage="${1:-}"
shift || true

if [[ "${stage}" != "1" && "${stage}" != "2" ]]; then
  echo "用法: bash data/scripts/protein/10_run_reasoning_api.sh <stage: 1|2> [额外参数]"
  echo "示例: bash data/scripts/protein/10_run_reasoning_api.sh 1 --end 20"
  exit 1
fi

python data/scripts/protein/10_generate_reasoning.py --stage "${stage}" "$@"
