import pandas as pd
import random
from pyliftover import LiftOver
import os
import re

print("🚀 【Step 1B】: 引入最新 VEP 名册，执行双向冲突净化，生成终极大表...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAIN_FILE  = os.path.join(BASE_DIR, "reference", "hg19ToHg38.over.chain.gz")
GOF_EXCEL   = os.path.join(BASE_DIR, "raw", "gofcards_data_download.xlsx")
CLINVAR_CSV = os.path.join(BASE_DIR, "raw", "goflof_ClinVar_v062021.csv")
CLINVAR_VCF = os.path.join(BASE_DIR, "vep_results", "clinvar.vcf")
OUT_DIR     = os.path.join(BASE_DIR, "processed")

# 这次我们用的是你刚刚新鲜跑出来、绝对复现的 VEP 结果！
GOF_VEP_TXT = os.path.join(OUT_DIR, "step1_gof_vep.txt")
lo = LiftOver(CHAIN_FILE)

# ================= 1. 获取新鲜出炉的标准发病基因池 =================
standard_gof_genes = set()
symbol_index = -1

with open(GOF_VEP_TXT, 'r') as f:
    for line in f:
        # 跳过开头的双井号元数据
        if line.startswith('##'): 
            continue 
            
        parts = line.strip().split('\t')
        
        # 如果遇到表头行，动态锁定 SYMBOL 列的索引位置
        if line.startswith('#Uploaded_variation'):
            if 'SYMBOL' in parts:
                symbol_index = parts.index('SYMBOL')
            continue
            
        # 如果是数据行，直接精准提取那一列！
        if symbol_index != -1 and len(parts) > symbol_index:
            gene_symbol = parts[symbol_index].strip()
            if gene_symbol and gene_symbol != '-':
                standard_gof_genes.add(gene_symbol)

print(f"🧬 从最新 VEP 结果中加载了 {len(standard_gof_genes)} 个标准 HGNC 基因！")

# ================= 2. 收集所有坐标 (为对撞做准备) =================
raw_gof_dict = {}
raw_lof_dict = {}

# 2.1 收集 GOF
df_gofcards = pd.read_excel(GOF_EXCEL)
for _, row in df_gofcards.iterrows():
    chrom = str(row['chr']).replace('chr', '')
    res = lo.convert_coordinate(f"chr{chrom}", int(row['hg19start']))
    if res:
        cid = f"{chrom}_{res[0][1]}_{str(row['ref']).upper()}_{str(row['alt']).upper()}"
        raw_gof_dict[cid] = {'CHROM': chrom, 'POS': res[0][1], 'ID': cid, 'REF': str(row['ref']).upper(), 'ALT': str(row['alt']).upper(), 'Gene': str(row.iloc[1]).strip(), 'LABEL': 'GOF'}

df_clinvar = pd.read_csv(CLINVAR_CSV)
col_map = {'CHROM': 'Chromosome', 'POS': 'PositionVCF', 'REF': 'ReferenceAlleleVCF', 'ALT': 'AlternateAlleleVCF', 'GENE': 'GeneSymbol'}
df_clinvar.rename(columns=lambda x: col_map.get(x.upper(), x), inplace=True)

for _, row in df_clinvar[df_clinvar['LABEL'] == 'GOF'].iterrows():
    chrom = str(row['Chromosome']).replace('chr', '')
    res = lo.convert_coordinate(f"chr{chrom}", int(row['PositionVCF']))
    if res:
        cid = f"{chrom}_{res[0][1]}_{str(row['ReferenceAlleleVCF']).upper()}_{str(row['AlternateAlleleVCF']).upper()}"
        raw_gof_dict[cid] = {'CHROM': chrom, 'POS': res[0][1], 'ID': cid, 'REF': str(row['ReferenceAlleleVCF']).upper(), 'ALT': str(row['AlternateAlleleVCF']).upper(), 'Gene': str(row.get('GeneSymbol', '-')), 'LABEL': 'GOF'}

# 2.2 收集 LOF (基于标准基因)
for _, row in df_clinvar[df_clinvar['LABEL'] == 'LOF'].iterrows():
    gene = str(row.get('GeneSymbol', '-')).strip()
    if gene not in standard_gof_genes: continue
    
    chrom = str(row['Chromosome']).replace('chr', '')
    res = lo.convert_coordinate(f"chr{chrom}", int(row['PositionVCF']))
    if res:
        cid = f"{chrom}_{res[0][1]}_{str(row['ReferenceAlleleVCF']).upper()}_{str(row['AlternateAlleleVCF']).upper()}"
        raw_lof_dict[cid] = {'CHROM': chrom, 'POS': res[0][1], 'ID': cid, 'REF': str(row['ReferenceAlleleVCF']).upper(), 'ALT': str(row['AlternateAlleleVCF']).upper(), 'Gene': gene, 'LABEL': 'LOF'}

# ================= 3. 💥 终极对撞与双向净化 💥 =================
conflicts = set(raw_gof_dict.keys()).intersection(set(raw_lof_dict.keys()))
print(f"\n💥 发现争议冲突变异 {len(conflicts)} 条！正在执行双向剔除...")

master_records = []
# 加入净化后的 GOF
for cid, record in raw_gof_dict.items():
    if cid not in conflicts: master_records.append(record)
# 加入净化后的 LOF
for cid, record in raw_lof_dict.items():
    if cid not in conflicts: master_records.append(record)

print(f"✅ 净化完毕！保留绝对纯正 GOF: {len(raw_gof_dict) - len(conflicts)} 条 | 绝对纯正 LOF: {len(raw_lof_dict) - len(conflicts)} 条。")

# ================= 4. 收集 Neutral 军团 =================
neutral_pool = []
try:
    with open(CLINVAR_VCF, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split('\t')
            info = parts[7]
            if "CLNSIG=Benign" in info or "CLNSIG=Likely_benign" in info:
                gene_match = [p.replace('GENEINFO=', '').split(':')[0] for p in info.split(';') if p.startswith('GENEINFO=')]
                if not gene_match: continue
                gene = gene_match[0]
                
                if gene in standard_gof_genes:
                    if "nonsense" in info.lower() or "frameshift" in info.lower() or "splice" in info.lower(): continue
                    chrom = parts[0].replace('chr', '')
                    if len(parts[3]) == 1 and len(parts[4]) == 1:
                        cid = f"{chrom}_{parts[1]}_{parts[3]}_{parts[4]}"
                        # 确保不与已有的致病变异(GOF/LOF/冲突)重合
                        if cid not in raw_gof_dict and cid not in raw_lof_dict:
                            neutral_pool.append({'CHROM': chrom, 'POS': parts[1], 'ID': cid, 'REF': parts[3], 'ALT': parts[4], 'Gene': gene, 'LABEL': 'Neutral'})
    
    if len(neutral_pool) > 3000:
        random.seed(42)
        neutral_pool = random.sample(neutral_pool, 3000)
    master_records.extend(neutral_pool)
    print(f"🔵 抓取到高质量中性变异: {len(neutral_pool)} 条。")
except FileNotFoundError:
    pass
    
    
    
    
# ================= 🚀 核心修复：为终极大表进行严格排序 =================
def sort_key(x):
    chrom = str(x['CHROM'])
    pos = int(x['POS'])
    try:
        c_num = int(chrom)
    except ValueError:
        c_num = 999 
    return (c_num, chrom, pos)

master_records.sort(key=sort_key)
print("✅ 终极大表已按照染色体和物理位置严格排序！")
# =====================================================================

# 5. 输出终极大一统文件
df_master = pd.DataFrame(master_records)
# ... 下面保持原样 ...

# ================= 5. 输出终极大一统文件 =================
df_master = pd.DataFrame(master_records)
df_master.to_csv(os.path.join(OUT_DIR, "master_raw_all_labels.csv"), index=False)

vcf_out = os.path.join(OUT_DIR, "master_to_annotate.vcf")
with open(vcf_out, "w") as f:
    f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for _, row in df_master.iterrows():
        f.write(f"{row['CHROM']}\t{row['POS']}\t{row['ID']}\t{row['REF']}\t{row['ALT']}\t.\t.\tLABEL={row['LABEL']};GENE={row['Gene']}\n")

print(f"\n🎉 完美复现！终极无冲突数据集已生成 (共 {len(df_master)} 条)。")
print(f"👉 下一步：用这个 {vcf_out} 去跑最后一次全量 VEP 吧！")