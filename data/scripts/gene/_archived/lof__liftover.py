import pandas as pd
from pyliftover import LiftOver
import pysam

print("🚀 启动带 REF 校验的智能坐标升维引擎 (Smart LiftOver)...")

input_vcf = "lof_all_types_to_annotate.vcf" # 你的原始 hg19 数据
output_vcf = "lof_hg38_smart_validated.vcf"
fasta_path = "/gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta"

# 1. 加载工具
try:
    lo = LiftOver('hg19ToHg38.over.chain.gz')
    fasta = pysam.FastaFile(fasta_path)
    print("✅ 字典与基因组加载成功！")
except Exception as e:
    print(f"❌ 工具加载失败: {e}")
    exit()

converted_count = 0
dropped_count = 0

with open(input_vcf, 'r') as fin, open(output_vcf, 'w') as fout:
    for line in fin:
        if line.startswith('#'):
            fout.write(line)
            continue
            
        parts = line.strip().split('\t')
        chrom_raw = parts[0]
        pos_1based = int(parts[1])
        ref_allele = parts[3]
        
        chrom_query = chrom_raw if chrom_raw.startswith('chr') else f"chr{chrom_raw}"
        
        # 0-based 转换用于 pyliftover 和 pysam
        pos_0based = pos_1based - 1 
        
        # 核心：查询 hg38 坐标
        new_coords = lo.convert_coordinate(chrom_query, pos_0based)
        
        if new_coords and len(new_coords) > 0:
            new_chrom = new_coords[0][0]
            new_pos_0based = new_coords[0][1]
            
            # 终极校验：去 hg38 fasta 里查这个位置的真实碱基是什么！
            try:
                # fetch 取左闭右开区间
                hg38_ref = fasta.fetch(reference=new_chrom, start=new_pos_0based, end=new_pos_0based+1).upper()
            except:
                hg38_ref = "N"
            
            # 只有当 hg38 上的碱基和我们老 VCF 里的 REF 完全一样时，才保留！
            if hg38_ref == ref_allele.upper():
                new_chrom_clean = new_chrom.replace('chr', '')
                new_pos_1based = new_pos_0based + 1
                
                parts[0] = new_chrom_clean
                parts[1] = str(new_pos_1based)
                parts[2] = f"{new_chrom_clean}_{new_pos_1based}_{ref_allele}_{parts[4]}"
                
                fout.write('\t'.join(parts) + '\n')
                converted_count += 1
            else:
                # 坐标虽然转成功了，但碱基变了，这种数据喂给 VEP 也会死，直接丢弃
                dropped_count += 1
        else:
            dropped_count += 1

print("="*40)
print(f"🏆 智能升维完成！")
print(f"🟢 完美存活 (坐标&碱基双重匹配): {converted_count} 条")
print(f"🔴 抛弃 (坐标丢失或 REF 碱基冲突): {dropped_count} 条")
print(f"💾 请使用新文件去跑 VEP: {output_vcf}")
print("="*40)
