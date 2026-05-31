import pandas as pd
from pyliftover import LiftOver
import os

print("🚀 【Step 1A 完美排序版】: 提取双库 GOF，生成 VCF 准备获取标准基因名...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAIN_FILE  = os.path.join(BASE_DIR, "reference", "hg19ToHg38.over.chain.gz")
GOF_EXCEL   = os.path.join(BASE_DIR, "raw", "gofcards_data_download.xlsx")
CLINVAR_CSV = os.path.join(BASE_DIR, "raw", "goflof_ClinVar_v062021.csv")
OUT_DIR     = os.path.join(BASE_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

lo = LiftOver(CHAIN_FILE)
gof_coords_hg38 = set()
gof_records = []

# 1. 拿 gofcards
df_gofcards = pd.read_excel(GOF_EXCEL)
for _, row in df_gofcards.iterrows():
    chrom = str(row['chr']).replace('chr', '')
    res = lo.convert_coordinate(f"chr{chrom}", int(row['hg19start']))
    if res:
        cid = f"{chrom}_{res[0][1]}_{str(row['ref']).upper()}_{str(row['alt']).upper()}"
        if cid not in gof_coords_hg38:
            gof_coords_hg38.add(cid)
            gof_records.append({'CHROM': chrom, 'POS': res[0][1], 'ID': cid, 'REF': str(row['ref']).upper(), 'ALT': str(row['alt']).upper()})

# 2. 拿 ClinVar 的 GOF
df_clinvar = pd.read_csv(CLINVAR_CSV)
col_map = {'CHROM': 'Chromosome', 'POS': 'PositionVCF', 'REF': 'ReferenceAlleleVCF', 'ALT': 'AlternateAlleleVCF'}
df_clinvar.rename(columns=lambda x: col_map.get(x.upper(), x), inplace=True)
for _, row in df_clinvar[df_clinvar['LABEL'] == 'GOF'].iterrows():
    chrom = str(row['Chromosome']).replace('chr', '')
    res = lo.convert_coordinate(f"chr{chrom}", int(row['PositionVCF']))
    if res:
        cid = f"{chrom}_{res[0][1]}_{str(row['ReferenceAlleleVCF']).upper()}_{str(row['AlternateAlleleVCF']).upper()}"
        if cid not in gof_coords_hg38:
            gof_coords_hg38.add(cid)
            gof_records.append({'CHROM': chrom, 'POS': res[0][1], 'ID': cid, 'REF': row['ReferenceAlleleVCF'], 'ALT': row['AlternateAlleleVCF']})

print(f"✅ 成功融合获得 {len(gof_records)} 条独立 GOF。")

# ================= 🚀 核心修复：极其严谨的 VEP 排序逻辑 =================
def sort_key(x):
    chrom = str(x['CHROM'])
    pos = int(x['POS'])
    # 把染色体转成数字，确保 2 排在 10 前面；X, Y 等字母排在最后
    try:
        c_num = int(chrom)
    except ValueError:
        c_num = 999 
    return (c_num, chrom, pos)

gof_records.sort(key=sort_key)
print("✅ 数据已按照染色体和物理位置严格排序！")
# =====================================================================

# 3. 导出 VCF
vcf_out = os.path.join(OUT_DIR, "step1_only_gof.vcf")
with open(vcf_out, "w") as f:
    f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for r in gof_records:
        f.write(f"{r['CHROM']}\t{r['POS']}\t{r['ID']}\t{r['REF']}\t{r['ALT']}\t.\t.\t.\n")

pd.DataFrame([{'ID': c} for c in gof_coords_hg38]).to_csv(os.path.join(OUT_DIR, "cache_gof_coords.csv"), index=False)

print(f"👉 第一步完成！请去终端运行 VEP...")