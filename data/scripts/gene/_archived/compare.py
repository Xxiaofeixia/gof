import pandas as pd

print("🚀 启动【全变异类型保留】坐标级比对流水线...")

# 1. 读取已经跑完特征的 GOF 数据 (获取 574 个核心基因)
df_gof = pd.read_csv("bioreason_final_result.csv")
target_genes = set(df_gof[df_gof['Gene_Symbol'] != '-']['Gene_Symbol'].dropna().unique())

# 2. 读取 VCF，建立 坐标 -> GOF行号 的映射
coord_to_row = {}
gof_coords = set()
row_idx = 1
with open("to_annotate.vcf", "r") as f:
    for line in f:
        if not line.startswith("#"):
            parts = line.strip().split('\t')
            chrom = parts[0].replace('chr', '')
            # 构建绝对坐标身份证: chr_pos_ref_alt
            coord_id = f"{chrom}_{parts[1]}_{parts[3]}_{parts[4]}"
            coord_to_row[coord_id] = row_idx
            gof_coords.add(coord_id)
            row_idx += 1

print(f"✅ 加载了 {len(gof_coords)} 个 GOF 变异的高精度物理坐标。")

# 3. 载入并处理 ClinVar 数据
try:
    df_clinvar = pd.read_csv("goflof_ClinVar_v062021.csv")
except:
    df_clinvar = pd.read_csv("goflof_ClinVar_v062021.csv", sep='\t')

col_map = {'CHROM': 'Chromosome', 'POS': 'PositionVCF', 'REF': 'ReferenceAlleleVCF', 'ALT': 'AlternateAlleleVCF', 'GENE': 'GeneSymbol'}
df_clinvar.rename(columns=lambda x: col_map.get(x.upper(), x), inplace=True)

# 🚨 关键：只过滤 574 个目标基因，绝对不过滤变异类型（保留错义、无义、移码等）
df_clinvar = df_clinvar[df_clinvar['GeneSymbol'].isin(target_genes)].copy()

# 为 ClinVar 生成同样的坐标身份证
df_clinvar['Coord_ID'] = df_clinvar.apply(
    lambda r: f"{str(r['Chromosome']).replace('chr', '')}_{r['PositionVCF']}_{r['ReferenceAlleleVCF']}_{r['AlternateAlleleVCF']}", axis=1)

# 4. 对撞排查
overlaps = gof_coords.intersection(set(df_clinvar['Coord_ID']))
overlaps_in_clinvar = df_clinvar[df_clinvar['Coord_ID'].isin(overlaps)]
conflicts = overlaps_in_clinvar[overlaps_in_clinvar['LABEL'] == 'LOF']

print(f"🔍 发现 {len(overlaps)} 个变异在物理坐标上完全重合。")
print(f"⚠️ 其中严重冲突（GOF被错误标为LOF）有 {len(conflicts)} 条。")

# 5. 剔除冲突，生成最终纯净版 GOF
conflict_rows = [coord_to_row[cid] for cid in conflicts['Coord_ID']]
df_gof_strict = df_gof[~df_gof['Original_Row'].isin(conflict_rows)].copy()
df_gof_strict.to_csv("bioreason_gof_strict_all_types.csv", index=False)
print(f"💾 已保存去除冲突的干净 GOF 数据 (共 {len(df_gof_strict)} 条)")

# 6. 生成供 VEP 注释的全类型 ClinVar 列表
df_clinvar_clean = df_clinvar[~df_clinvar['Coord_ID'].isin(overlaps)].copy()
df_clinvar_clean.to_csv("clinvar_lof_metadata.csv", index=False) # 存一份元数据备用

with open("lof_all_types_to_annotate.vcf", "w") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for _, row in df_clinvar_clean.iterrows():
        chrom = str(row['Chromosome']).replace('chr', '')
        # 巧妙操作：把 Coord_ID 写入 VCF 的 ID 列，方便跑完 VEP 后精确匹配回来
        cid = row['Coord_ID'] 
        f.write(f"{chrom}\t{row['PositionVCF']}\t{cid}\t{row['ReferenceAlleleVCF']}\t{row['AlternateAlleleVCF']}\t.\t.\tLABEL={row['LABEL']}\n")

print(f"🏆 已生成全类型背景变异 VCF：lof_all_types_to_annotate.vcf (共 {len(df_clinvar_clean)} 条)")
