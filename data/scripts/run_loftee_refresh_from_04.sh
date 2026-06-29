#!/bin/bash
# 从第 04 步 VEP+LOFTEE 开始，刷新到第 07 步 BioReason prompt。
# 用法：
#   bash data/scripts/run_loftee_refresh_from_04.sh              # 自动提交 04_vep_for_gene.sh
#   bash data/scripts/run_loftee_refresh_from_04.sh <VEP_JOBID>  # 复用已经提交的 VEP 作业

set -eo pipefail

PROJECT_DIR="/gpfs/hpc/home/lijc/mapengtao/gof"
PROCESSED_DIR="${PROJECT_DIR}/data/processed"
GENE_SCRIPT_DIR="${PROJECT_DIR}/data/scripts/gene"
PROTEIN_SCRIPT_DIR="${PROJECT_DIR}/data/scripts/protein"
VEP_JOB_ID_ARG="${1:-}"

source /gpfs/hpc/home/public/jclabadmin/software/anaconda3/bin/activate gof

log_step() {
    echo
    echo "[$(date '+%F %T')] $*"
}

latest_state() {
    local job_id="$1"
    sacct -j "${job_id}" --format=JobID,State -P -n \
        | awk -F'|' -v id="${job_id}" '$1 == id {print $2}' \
        | tail -1 \
        | cut -d' ' -f1
}

wait_for_job() {
    local job_id="$1"
    local label="$2"
    local state=""

    log_step "等待 ${label} 作业 ${job_id} 完成"
    while squeue -j "${job_id}" -h >/dev/null 2>&1 && [ -n "$(squeue -j "${job_id}" -h)" ]; do
        squeue -j "${job_id}" -o '%i|%P|%j|%T|%M|%l|%C|%m|%R' || true
        sleep 60
    done

    state="$(latest_state "${job_id}")"
    log_step "${label} 作业 ${job_id} 终态: ${state}"
    if [ "${state}" != "COMPLETED" ]; then
        echo "错误：${label} 作业未成功完成，停止后续流程。" >&2
        sacct -j "${job_id}" --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ReqMem,NodeList -P || true
        exit 1
    fi
}

run_python_step() {
    local label="$1"
    local workdir="$2"
    local script="$3"
    local logfile="$4"

    log_step "运行 ${label}: ${script}"
    cd "${workdir}"
    python ${script} > "${logfile}" 2>&1
    tail -40 "${logfile}"
}

if [ -n "${VEP_JOB_ID_ARG}" ]; then
    VEP_JOB_ID="${VEP_JOB_ID_ARG}"
else
    log_step "提交第 04 步 VEP+LOFTEE"
    VEP_JOB_ID="$(cd "${PROJECT_DIR}" && sbatch "${GENE_SCRIPT_DIR}/04_vep_for_gene.sh" | awk '{print $4}')"
    log_step "VEP JobID: ${VEP_JOB_ID}"
fi

wait_for_job "${VEP_JOB_ID}" "04 VEP+LOFTEE"

run_python_step "05 提取 VEP/LOFTEE 注释和 DNA 双序列" \
    "${GENE_SCRIPT_DIR}" "05_extract_gene_feature.py" \
    "${PROCESSED_DIR}/05_extract_gene_feature.log"

run_python_step "protein 01 融合 gnomAD/ClinGen 基因背景" \
    "${PROTEIN_SCRIPT_DIR}" "01_add_db_features.py" \
    "${PROCESSED_DIR}/protein_01_add_db_features.log"

run_python_step "protein 02 构建 UniProt 映射并补齐 AlphaFold PDB 缓存" \
    "${PROTEIN_SCRIPT_DIR}" "02_download_alphafold_pdb.py" \
    "${PROCESSED_DIR}/protein_02_download_alphafold_pdb.log"

run_python_step "protein 03 DSSP/RSA 二级结构注释" \
    "${PROTEIN_SCRIPT_DIR}" "03_calc_dssp_rsa.py" \
    "${PROCESSED_DIR}/protein_03_calc_dssp_rsa.log"

log_step "提交 protein 04 ESM GPU 作业"
ESM_JOB_ID="$(cd "${PROTEIN_SCRIPT_DIR}" && sbatch "04_run_esm_gpu.sh" | awk '{print $4}')"
log_step "ESM JobID: ${ESM_JOB_ID}"
wait_for_job "${ESM_JOB_ID}" "protein 04 ESM"

run_python_step "protein 05 空间密度" \
    "${PROTEIN_SCRIPT_DIR}" "05_calc_spatial_density.py" \
    "${PROCESSED_DIR}/protein_05_calc_spatial_density.log"

run_python_step "protein 06 氨基酸生化特征" \
    "${PROTEIN_SCRIPT_DIR}" "06_calc_biochemical_features.py" \
    "${PROCESSED_DIR}/protein_06_calc_biochemical_features.log"

run_python_step "protein 06b UniProt/LOFTEE 推理上下文特征" \
    "${PROTEIN_SCRIPT_DIR}" "06b_add_reasoning_context_features.py" \
    "${PROCESSED_DIR}/protein_06b_add_reasoning_context_features.log"

run_python_step "protein 00 两阶段显著性检验" \
    "${PROTEIN_SCRIPT_DIR}" "00_statistical_feature_selection.py" \
    "${PROCESSED_DIR}/protein_00_statistical_feature_selection.log"

run_python_step "protein 07 阶段一 prompt" \
    "${PROTEIN_SCRIPT_DIR}" "07_format_bioreason_prompt.py --stage 1" \
    "${PROCESSED_DIR}/protein_07_stage1_format.log"

run_python_step "protein 07 阶段二 prompt" \
    "${PROTEIN_SCRIPT_DIR}" "07_format_bioreason_prompt.py --stage 2" \
    "${PROCESSED_DIR}/protein_07_stage2_format.log"

log_step "流程完成。正式输出已覆盖到 data/processed。"
