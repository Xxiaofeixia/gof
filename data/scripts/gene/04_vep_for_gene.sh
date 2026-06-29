#!/bin/bash
#SBATCH --job-name=VEP_GOF_Main
#SBATCH --partition=FAT1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/04_vep_for_gene_%j.log

set -eo pipefail

# 本脚本只负责第 04 步：对当前 01/03 生成的 VCF 进行离线 VEP 注释。
# 输入：
#   data/processed/02_pathogenic_to_annotate.vcf
#   data/processed/03_neutral_to_annotate.vcf
# 输出：
#   data/processed/04_pathogenic_vep_output.txt
#   data/processed/04_neutral_vep_output.txt

source /gpfs/hpc/home/public/jclabadmin/software/anaconda3/bin/activate
conda activate gof

PROJECT_DIR=/gpfs/hpc/home/lijc/mapengtao/gof
PROCESSED_DIR=${PROJECT_DIR}/data/processed
VEP_CACHE=/gpfs/hpc/home/public/jclabadmin/data/vep/vep_110
FASTA=/gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta
DBNSFP=${VEP_CACHE}/dbNSFP4.7a_grch38.gz
LOFTEE_DIR=${VEP_CACHE}/Plugins/loftee-grch38
LOFTEE_ANCESTOR=${VEP_CACHE}/ancestor/human_ancestor.fa.gz
LOFTEE_GERP=${LOFTEE_DIR}/gerp_conservation_scores.homo_sapiens.GRCh38.bw
LOFTEE_CONSERVATION=${VEP_CACHE}/loftee.sql
LOFTEE_SHIM=${PROJECT_DIR}/data/reference/loftee_perl_shim

PATHOGENIC_VCF=${PROCESSED_DIR}/02_pathogenic_to_annotate.vcf
NEUTRAL_VCF=${PROCESSED_DIR}/03_neutral_to_annotate.vcf
PATHOGENIC_OUT=${PROCESSED_DIR}/04_pathogenic_vep_output.txt
NEUTRAL_OUT=${PROCESSED_DIR}/04_neutral_vep_output.txt

# LOFTEE 的 Perl 脚本互相 require，因此需要把插件目录加入 PERL5LIB。
# 当前环境的 BioPerl 缺少旧版 Bio::Perl 兼容模块，项目内 shim 只补插件加载依赖。
export PERL5LIB="${LOFTEE_SHIM}:${LOFTEE_DIR}:${PERL5LIB:-}"

echo "[$(date)] 开始第 04 步 VEP 注释"
echo "致病 VCF: ${PATHOGENIC_VCF}"
echo "中性 VCF: ${NEUTRAL_VCF}"

echo "[$(date)] 注释致病 GOF/LOF 变异..."
vep -i "${PATHOGENIC_VCF}" \
    -o "${PATHOGENIC_OUT}" \
    --force_overwrite --tab \
    --assembly GRCh38 --offline \
    --dir_cache "${VEP_CACHE}" \
    --merged \
    --fasta "${FASTA}" \
    --everything \
    --fork 8 \
    --dir_plugins "${LOFTEE_DIR}" \
    --plugin LoF,loftee_path:"${LOFTEE_DIR}",human_ancestor_fa:"${LOFTEE_ANCESTOR}",conservation_file:"${LOFTEE_CONSERVATION}",use_gerp_end_trunc:0 \
    --plugin dbNSFP,"${DBNSFP}",REVEL_score,CADD_phred,AlphaMissense_score,GERP++_RS,phyloP100way_vertebrate,MutPred_score,SpliceAI_pred_DS_AG,SpliceAI_pred_DS_AL,SpliceAI_pred_DS_DG,SpliceAI_pred_DS_DL

echo "[$(date)] 注释 gnomAD neutral 变异..."
vep -i "${NEUTRAL_VCF}" \
    -o "${NEUTRAL_OUT}" \
    --force_overwrite --tab \
    --assembly GRCh38 --offline \
    --dir_cache "${VEP_CACHE}" \
    --merged \
    --fasta "${FASTA}" \
    --everything \
    --fork 8 \
    --dir_plugins "${LOFTEE_DIR}" \
    --plugin LoF,loftee_path:"${LOFTEE_DIR}",human_ancestor_fa:"${LOFTEE_ANCESTOR}",conservation_file:"${LOFTEE_CONSERVATION}",use_gerp_end_trunc:0 \
    --plugin dbNSFP,"${DBNSFP}",REVEL_score,CADD_phred,AlphaMissense_score,GERP++_RS,phyloP100way_vertebrate,MutPred_score,SpliceAI_pred_DS_AG,SpliceAI_pred_DS_AL,SpliceAI_pred_DS_DG,SpliceAI_pred_DS_DL

echo "[$(date)] 第 04 步 VEP 注释完成"
