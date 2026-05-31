import pandas as pd
from pyliftover import LiftOver

import os

print("🔍 启动【GOF 双库重合度侦探程序】...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAIN_FILE  = os.path.join(BASE_DIR, "reference", "hg19ToHg38.over.chain.gz")
GOF_EXCEL   = os.path.join(BASE_DIR, "raw", "gofcards_data_download.xlsx")
CLINVAR_CSV = os.path.join(BASE_DIR, "raw", "goflof_ClinVar_v062021.csv")

lo = LiftOver(CHAIN_FILE)

# ================= 1. 提取 gofcards 的 hg38 坐标集合 =================
gofcards_set = set()
df_gofcards = pd.read_excel(GOF_EXCEL)
for i, row in df_gofcards.iterrows():
    chrom = str(row['chr']).replace('chr', '')
    res = lo.convert_coordinate(f"chr{chrom}", int(row['hg19start']))
    if res:
        cid = f"{chrom}_{res[0][1]}_{str(row['ref']).upper()}_{str(row['alt']).upper()}"
        gofcards_set.add(cid)

print(f"📘 gofcards 成功转换为 hg38 的纯变异数量: {len(gofcards_set)}")

# ================= 2. 提取 ClinVar GOF 的 hg38 坐标集合 =================
clinvar_gof_set = set()
df_clinvar = pd.read_csv(CLINVAR_CSV)
col_map = {'CHROM': 'Chromosome', 'POS': 'PositionVCF', 'REF': 'ReferenceAlleleVCF', 'ALT': 'AlternateAlleleVCF'}
df_clinvar.rename(columns=lambda x: col_map.get(x.upper(), x), inplace=True)

for i, row in df_clinvar[df_clinvar['LABEL'] == 'GOF'].iterrows():
    chrom = str(row['Chromosome']).replace('chr', '')
    res = lo.convert_coordinate(f"chr{chrom}", int(row['PositionVCF']))
    if res:
        cid = f"{chrom}_{res[0][1]}_{str(row['ReferenceAlleleVCF']).upper()}_{str(row['AlternateAlleleVCF']).upper()}"
        clinvar_gof_set.add(cid)

print(f"📕 ClinVar(GOF) 成功转换为 hg38 的纯变异数量: {len(clinvar_gof_set)}")

# ================= 3. 计算交集与去重 =================
overlap = gofcards_set.intersection(clinvar_gof_set)
only_gofcards = gofcards_set - clinvar_gof_set
only_clinvar = clinvar_gof_set - gofcards_set
total_unique = gofcards_set.union(clinvar_gof_set)

print("\n================ 📊 终极盘点报告 ================")
print(f"🤝 双方完全重合的变异 (英雄所见略同): {len(overlap)} 条")
print(f"🛡️ 只有 gofcards 收录的独家变异: {len(only_gofcards)} 条")
print(f"🗡️ 只有 ClinVar 收录的独家变异 (这是你的神补刀): {len(only_clinvar)} 条")
print(f"🏆 最终去重后，喂给大模型的无敌正样本总数: {len(total_unique)} 条 (也就是你的 2394 条！)")
print("=================================================")
