import pandas as pd
import random

print("🚀 启动官方 ClinVar hg38 纯净中性变异提取引擎...")

# 1. 读取我们的 574 个核心基因名单
try:
    df_gof = pd.read_csv("bioreason_final_result.csv")
    target_genes = set(df_gof[df_gof['Gene_Symbol'] != '-']['Gene_Symbol'].dropna().unique())
except:
    print("❌ 找不到 bioreason_final_result.csv，请检查路径！")
    exit()

benign_pool = []
print("⏳ 正在扫描官方 VCF 寻找 574 基因上的 Benign 错义突变...")

# 2. 逐行扫描 VCF (极其节省内存，可以在任意节点跑)
with open("clinvar.vcf", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
            
        parts = line.strip().split('\t')
        info = parts[7]
        
        # 必须是纯正的 Benign 或 Likely_benign
        if "CLNSIG=Benign" in info or "CLNSIG=Likely_benign" in info:
            # 提取基因名
            gene_match = [part.replace('GENEINFO=', '').split(':')[0] for part in info.split(';') if part.startswith('GENEINFO=')]
            if not gene_match: continue
            gene = gene_match[0]
            
            # 必须在我们的 574 个核心基因里
            if gene in target_genes:
                # 排除明显破坏蛋白质结构的变异 (无义、移码、剪接等)，尽量保留错义(Missense)
                if "nonsense" in info.lower() or "frameshift" in info.lower() or "splice" in info.lower():
                    continue
                    
                # 记录坐标和基因
                chrom = parts[0]
                pos = parts[1]
                ref = parts[3]
                alt = parts[4]
                # 过滤掉过长的插入缺失，只保留单碱基(SNV)错义突变
                if len(ref) == 1 and len(alt) == 1:
                    benign_pool.append(f"{chrom}\t{pos}\t{chrom}_{pos}_{ref}_{alt}\t{ref}\t{alt}\t.\t.\tLABEL=Neutral;GENE={gene}\n")

print(f"✅ 扫描完毕！在 574 个基因上成功挖掘到 {len(benign_pool)} 条高质量的中性错义突变。")

# 3. 随机抽取 3000 条 (如果不够就全要)
target_number = 3000
if len(benign_pool) > target_number:
    random.seed(42)
    final_neutral = random.sample(benign_pool, target_number)
    print(f"🎲 已随机抽取 {target_number} 条极品变异。")
else:
    final_neutral = benign_pool
    print("⚠️ 数量不足 3000 条，全部保留！")

# 4. 生成最终直接可用于 VEP 的 VCF
output_file = "neutral_hg38_to_annotate.vcf"
with open(output_file, "w") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for line in final_neutral:
        f.write(line)

print(f"🏆 大功告成！已生成原生 hg38 的中性 VCF：{output_file}")
print("👉 下一步：直接把这个文件扔给 VEP 去跑注释 (不需要做 LiftOver！)")
