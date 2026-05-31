import pandas as pd
import re

print("🚀 正在解析 VEP 结果并进行【变异类型全维度统计】...")

# 1. 从 VEP 提取 LOF 特征
vep_file = "lof_vep_output.txt"
lof_features = {}

with open(vep_file, "r") as f:
    for line in f:
        if not line.startswith("#"):
            parts = line.strip().split('\t')
            # 这里的 parts[0] 就是我们在第一步写入的 Coord_ID
            coord_id = parts[0] 
            aa = parts[10] if len(parts) > 10 else "-"
            extra = parts[13] if len(parts) > 13 else ""
            
            # 正则提取专家分数
            am_match = re.search(r'AlphaMissense_score=([\d\.]+)', extra)
            revel_match = re.search(r'REVEL_score=([\d\.]+)', extra)
            am = am_match.group(1) if am_match else "-"
            revel = revel_match.group(1) if revel_match else "-"
            
            # 去重：VEP 可能会输出多行转录本，保留含有氨基酸改变的那行
            if coord_id not in lof_features or aa != '-':
                lof_features[coord_id] = {'Amino_acids': aa, 'AlphaMissense': am, 'REVEL_score': revel}

# 2. 将特征拼回 LOF 元数据
df_lof = pd.read_csv("clinvar_lof_metadata.csv")
df_lof['Amino_acids'] = df_lof['Coord_ID'].map(lambda x: lof_features.get(x, {}).get('Amino_acids', '-'))
df_lof['AlphaMissense'] = df_lof['Coord_ID'].map(lambda x: lof_features.get(x, {}).get('AlphaMissense', '-'))
df_lof['REVEL_score'] = df_lof['Coord_ID'].map(lambda x: lof_features.get(x, {}).get('REVEL_score', '-'))

# 3. 读取严格版 GOF
df_gof = pd.read_csv("bioreason_gof_strict_all_types.csv")

# =================定义分类法则=================
def classify_mut(aa):
    aa = str(aa)
    if aa == '-' or aa == '' or aa == 'nan': return 'Non-coding/Unknown (非编码/未知)'
    if '*' in aa or 'Ter' in aa: return 'Nonsense (无义突变)'
    if 'fs' in aa: return 'Frameshift (移码突变)'
    if 'del' in aa or 'dup' in aa or 'ins' in aa: return 'InDel (框内缺失/插入)'
    if re.match(r'^[A-Z][a-z]{0,2}/[A-Z][a-z]{0,2}$', aa): return 'Missense (错义突变)'
    return 'Other (其他)'

df_gof['Mut_Type'] = df_gof['Amino_acids'].apply(classify_mut)
df_lof['Mut_Type'] = df_lof['Amino_acids'].apply(classify_mut)

# =================打印极具科研价值的大盘统计=================
print(f"\n========== 🟢 GOF 变异类型分布 (总计 {len(df_gof)} 条) ==========")
print(df_gof['Mut_Type'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
print(df_gof['Mut_Type'].value_counts())

print(f"\n========== 🔴 LOF/Neutral 变异类型分布 (总计 {len(df_lof)} 条) ==========")
print(df_lof['Mut_Type'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
print(df_lof['Mut_Type'].value_counts())

# =================数据大一统=================
# 提取共有的核心列进行合并
df_gof['LABEL'] = 'GOF' # 确保有标签
cols_to_keep_gof = ['Gene_Symbol', 'Amino_acids', 'Mut_Type', 'AlphaMissense', 'REVEL_score', 'LABEL']
cols_to_keep_lof = ['GeneSymbol', 'Amino_acids', 'Mut_Type', 'AlphaMissense', 'REVEL_score', 'LABEL']

df_gof_export = df_gof[cols_to_keep_gof].copy()
df_lof_export = df_lof[cols_to_keep_lof].rename(columns={'GeneSymbol': 'Gene_Symbol'}).copy()

df_final = pd.concat([df_gof_export, df_lof_export], ignore_index=True)
df_final.to_csv("merged_gof_lof_all_features.csv", index=False)
print("\n💾 已成功生成包含所有特征与突变类型注释的终极大表：merged_gof_lof_all_features.csv")
