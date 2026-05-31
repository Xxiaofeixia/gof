#!/bin/bash
#SBATCH --job-name=vep_mini_gof
#SBATCH --partition=CU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=vep_mini_%j.log
#SBATCH --error=vep_mini_err_%j.log

source /gpfs/hpc/home/public/jclabadmin/software/anaconda3/bin/activate
conda activate gof

echo "🚀 开始运行迷你版 VEP (获取标准基因名)..."

vep -i /gpfs/hpc/home/lijc/mapengtao/gof/data/processed/step1_only_gof.vcf \
    -o /gpfs/hpc/home/lijc/mapengtao/gof/data/processed/step1_gof_vep.txt \
    --force_overwrite --tab --assembly GRCh38 --offline \
    --dir_cache /gpfs/hpc/home/public/jclabadmin/data/vep/vep_110/ \
    --merged \
    --fasta /gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta \
    --symbol --fork 8

echo "🎉 迷你版 VEP 运行完毕！"