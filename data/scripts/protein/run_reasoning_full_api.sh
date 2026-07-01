#!/usr/bin/env bash
set -eo pipefail

cd /gpfs/hpc/home/lijc/mapengtao/gof

bash data/scripts/protein/10_run_reasoning_api.sh 1 \
  --workers 5 \
  --max_retries 5 \
  > data/processed/protein_10_stage1_full.log 2>&1

bash data/scripts/protein/10_run_reasoning_api.sh 2 \
  --workers 5 \
  --max_retries 5 \
  > data/processed/protein_10_stage2_full.log 2>&1

echo DONE > data/processed/protein_10_full.done
