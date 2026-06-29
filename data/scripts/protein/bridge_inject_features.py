"""
桥接脚本: 把 VEP 新特征直接注入最终 CSV，跳过整个流水线重跑
=============================================================

读取 VEP 输出 + 现有 05_BIOREASON_with_Topology.csv，
计算/提取 6 个新特征，输出 06_BIOREASON_with_Biochem.csv

用法:
  python bridge_inject_features.py
"""

import pandas as pd
import numpy as np
import io
import os

print("🚀 桥接注入脚本 — 从 VEP 直接提取新特征...")

# ==========================================
# 1. 路径配置
# ==========================================
VEP_TXT = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/master_vep_output.txt"
INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/05_BIOREASON_with_Topology.csv"
OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/06_BIOREASON_with_Biochem.csv"

# ==========================================
# 2. 氨基酸理化性质表
# ==========================================
AA_PI = {
    'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77,
    'C': 5.07, 'E': 3.22, 'Q': 5.65, 'G': 5.97,
    'H': 7.59, 'I': 6.02, 'L': 5.98, 'K': 9.74,
    'M': 5.74, 'F': 5.48, 'P': 6.30, 'S': 5.68,
    'T': 5.60, 'W': 5.89, 'Y': 5.66, 'V': 5.96,
}

AA_MW = {
    'A': 89.09,  'R': 174.20, 'N': 132.12, 'D': 133.10,
    'C': 121.15, 'E': 147.13, 'Q': 146.15, 'G': 75.07,
    'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19,
    'M': 149.21, 'F': 165.19, 'P': 115.13, 'S': 105.09,
    'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
}

# ==========================================
# 3. 从 VEP 提取新特征 → ID → feature 字典
# ==========================================
print("⏳ 解析 VEP 输出...")
with open(VEP_TXT, 'r') as f:
    lines = [line for line in f if not line.startswith('##')]

df_vep = pd.read_csv(io.StringIO(''.join(lines)), sep='\t', low_memory=False)
df_vep.rename(columns={'#Uploaded_variation': 'ID'}, inplace=True)

# 去重（同 05 脚本逻辑：CANONICAL 优先，有 AA 变化优先）
CONSEQUENCE_RANK = {
    'stop_gained': 1, 'frameshift_variant': 2,
    'splice_acceptor_variant': 3, 'splice_donor_variant': 4,
    'start_lost': 5, 'stop_lost': 6,
    'missense_variant': 7, 'inframe_deletion': 8, 'inframe_insertion': 9,
    'protein_altering_variant': 10, 'coding_sequence_variant': 11,
    'synonymous_variant': 12, 'splice_region_variant': 13,
    '5_prime_UTR_variant': 14, '3_prime_UTR_variant': 15,
    'intron_variant': 16, 'non_coding_transcript_exon_variant': 17,
    'upstream_gene_variant': 18, 'downstream_gene_variant': 19,
    'intergenic_variant': 20
}

sort_keys = []
for _, row in df_vep.iterrows():
    aa = str(row.get('Amino_acids', '-'))
    has_aa = 0 if (aa in ('-', '', 'None', 'nan') or pd.isna(row.get('Amino_acids'))) else 1
    is_canon = 0 if str(row.get('CANONICAL', '')).upper() == 'YES' else 1
    conseq = str(row.get('Consequence', '')).split(',')[0].strip()
    conseq_rank = CONSEQUENCE_RANK.get(conseq, 50)
    sort_keys.append((conseq_rank, is_canon, -has_aa))

df_vep['_sort_key'] = sort_keys
df_vep = df_vep.sort_values('_sort_key').drop_duplicates(subset=['ID'], keep='first').copy()

# 构建 ID → 特征 映射
print("⏳ 计算 SpliceAI_DS_max / MutPred_score / in_last_exon...")

# SpliceAI: dbNSFP 4.7a 没有 SpliceAI 列，全部置 NaN（以后换 dbNSFP 4.8+ 再启用）
spliceai_dict = {}

# MutPred_score（格式: '-', '0.887,0.887,' → 取逗号分隔的最大值）
def _parse_mutpred(val):
    s = str(val).strip()
    if s in ('-', '', 'nan', 'None', 'invalid_field'):
        return float('nan')
    nums = []
    for p in s.split(','):
        p = p.strip()
        if p and p != '-':
            try:
                nums.append(float(p))
            except ValueError:
                continue
    return max(nums) if nums else float('nan')

if 'MutPred_score' in df_vep.columns:
    mutpred_dict = dict(zip(df_vep['ID'], df_vep['MutPred_score'].apply(_parse_mutpred)))
else:
    mutpred_dict = {}


# in_last_exon (from EXON field: "3/10")
if 'EXON' in df_vep.columns:
    def parse_in_last_exon(s):
        try:
            cur, total = str(s).strip().split('/')
            return 1 if int(cur) == int(total) else 0
        except (ValueError, AttributeError):
            return 0
    ex_dict = dict(zip(df_vep['ID'], df_vep['EXON'].apply(parse_in_last_exon)))
else:
    ex_dict = {}

# Protein_position (already in VEP, but let's also extract)
if 'Protein_position' in df_vep.columns:
    pp_dict = dict(zip(df_vep['ID'], df_vep['Protein_position'].fillna('Unknown')))
else:
    pp_dict = {}

print(f"   VEP 特征提取完毕: {len(spliceai_dict)} 条 SpliceAI, {len(mutpred_dict)} 条 MutPred, {len(ex_dict)} 条 EXON")

# ==========================================
# 4. 加载现有 CSV
# ==========================================
print("⏳ 加载现有大表...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"   共 {len(df)} 条记录")

# ==========================================
# 5. 注入 VEP 特征
# ==========================================
print("⏳ 注入 VEP 新特征...")

df['SpliceAI_DS_max'] = df['ID'].map(spliceai_dict).fillna(0.0)
df['MutPred_score'] = df['ID'].map(mutpred_dict)  # NaN 表示dbNSFP无数据，不像SpliceAI填0
df['in_last_exon'] = df['ID'].map(ex_dict).fillna(0).astype(int)

# Protein_position: 只在缺失时补（优先保留已有数据）
if 'Protein_position' not in df.columns:
    df['Protein_position'] = df['ID'].map(pp_dict).fillna('Unknown')

# ==========================================
# 6. 计算生化特征
# ==========================================
print("⏳ 计算 Isoelectric_diff / Molecular_weight...")

def parse_aa(aa_str):
    s = str(aa_str).strip()
    if pd.isna(aa_str) or s in ('', '-', 'None', 'nan'):
        return None, None
    if '/' not in s:
        return None, None
    parts = s.split('/')
    wt = parts[0].strip()
    mt = parts[-1].strip()
    if '*' in wt or '*' in mt or len(wt) == 0 or len(mt) == 0:
        return None, None
    return wt[0].upper(), mt[0].upper()

iso_diffs, mw_diffs = [], []
valid = 0
for _, row in df.iterrows():
    wt, mt = parse_aa(row.get('Amino_acids', ''))
    if wt and mt:
        valid += 1
        iso_diffs.append(round(abs(AA_PI.get(wt, 0) - AA_PI.get(mt, 0)), 4))
        mw_diffs.append(round(abs(AA_MW.get(wt, 0) - AA_MW.get(mt, 0)), 4))
    else:
        iso_diffs.append(float('nan'))
        mw_diffs.append(float('nan'))

df['Isoelectric_diff'] = iso_diffs
df['Molecular_weight'] = mw_diffs
print(f"   有效氨基酸变化: {valid}/{len(df)}")

# ==========================================
# 7. 输出
# ==========================================
df.to_csv(OUTPUT_CSV, index=False)

new_cols = ['SpliceAI_DS_max', 'MutPred_score', 'in_last_exon', 'Protein_position', 'Isoelectric_diff', 'Molecular_weight']
print("\n" + "=" * 70)
print("🏆 桥接注入完毕！新特征汇总:")
for c in new_cols:
    status = "✓" if c in df.columns else "✗"
    non_zero = int((df[c].fillna(0) != 0).sum()) if c in df.columns else 0
    print(f"   {status} {c}: 非零/非空={non_zero}")
print(f"💾 保存至: {OUTPUT_CSV}")
print("=" * 70)
print("\n👉 下一步: python 00_statistical_feature_selection.py && python 07_format_bioreason_prompt.py --stage 2")
