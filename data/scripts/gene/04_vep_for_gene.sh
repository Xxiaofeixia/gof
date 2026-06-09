#!/bin/bash
#SBATCH --job-name=VEP_Master_Full
#SBATCH --partition=FAT1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=vep_master_%j.log

# 1. 激活环境
source /gpfs/hpc/home/public/jclabadmin/software/anaconda3/bin/activate
conda activate gof

echo "🚀 开始为 6467 条终极大一统数据进行满血版 VEP 注释..."

# 2. 运行满血版 VEP (一次性注释所有 GOF, LOF, Neutral)
vep -i /gpfs/hpc/home/lijc/mapengtao/gof/data/processed/master_to_annotate.vcf \
    -o /gpfs/hpc/home/lijc/mapengtao/gof/data/processed/master_vep_output.txt \
    --force_overwrite --tab \
    --assembly GRCh38 --offline \
    --dir_cache /gpfs/hpc/home/public/jclabadmin/data/vep/vep_110/ \
    --merged \
    --fasta /gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta \
    --everything \
    --fork 8 \
    --plugin dbNSFP,/gpfs/hpc/home/public/jclabadmin/data/vep/vep_110/dbNSFP4.7a_grch38.gz,REVEL_score,CADD_phred,AlphaMissense_score,GERP++_RS,phyloP100way_vertebrate,MutPred_score,SpliceAI_pred_DS_AG,SpliceAI_pred_DS_AL,SpliceAI_pred_DS_DG,SpliceAI_pred_DS_DL

echo "🎉 终极 VEP 数据特征库已全部构建完毕！"
