"""
Protein Pipeline 06 — 氨基酸生化特征计算
=========================================

从 Amino_acids 字段计算:
  - Isoelectric_diff: 野生型与突变型氨基酸的等电点(pI)差值绝对值
  - Molecular_weight:  野生型与突变型氨基酸的分子量差值绝对值

输入: 05_BIOREASON_with_Topology.csv
输出: 06_BIOREASON_with_Biochem.csv
"""

import pandas as pd
import numpy as np
import os

print("🚀 [Protein Pipeline 06] 启动氨基酸生化特征 (Isoelectric_diff, Molecular_weight) 计算引擎...")

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/05_BIOREASON_with_Topology.csv"
OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/06_BIOREASON_with_Biochem.csv"

# ==========================================
# 2. 氨基酸理化性质表
# ==========================================
# 等电点 pI (Zimmerman et al., 1968)
AA_PI = {
    'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77,
    'C': 5.07, 'E': 3.22, 'Q': 5.65, 'G': 5.97,
    'H': 7.59, 'I': 6.02, 'L': 5.98, 'K': 9.74,
    'M': 5.74, 'F': 5.48, 'P': 6.30, 'S': 5.68,
    'T': 5.60, 'W': 5.89, 'Y': 5.66, 'V': 5.96,
}

# 分子量 Da (monoisotopic)
AA_MW = {
    'A': 89.09,  'R': 174.20, 'N': 132.12, 'D': 133.10,
    'C': 121.15, 'E': 147.13, 'Q': 146.15, 'G': 75.07,
    'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19,
    'M': 149.21, 'F': 165.19, 'P': 115.13, 'S': 105.09,
    'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
}

# ==========================================
# 3. 解析氨基酸变化
# ==========================================
def parse_aa_change(aa_str):
    """解析 'R/H' 格式的氨基酸变化，返回 (wt_aa, mt_aa) 或 (None, None)"""
    s = str(aa_str).strip()
    if pd.isna(aa_str) or s in ('', '-', 'None', 'nan'):
        return None, None
    if '/' not in s:
        return None, None
    parts = s.split('/')
    wt = parts[0].strip()
    mt = parts[-1].strip()
    # 排除终止密码子
    if '*' in wt or '*' in mt or len(wt) == 0 or len(mt) == 0:
        return None, None
    return wt[0].upper(), mt[0].upper()

def calc_diff(wt, mt, table):
    """计算 |value_wt - value_mt|，无效时返回 NaN（format 阶段会显示为 Unknown）"""
    if wt is None or mt is None:
        return float('nan')
    v_wt = table.get(wt)
    v_mt = table.get(mt)
    if v_wt is None or v_mt is None:
        return float('nan')
    return round(abs(v_wt - v_mt), 4)

# ==========================================
# 4. 加载数据
# ==========================================
print("⏳ 加载带有拓扑密度的大表...")
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"✅ 读取成功，共 {len(df)} 条")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# ==========================================
# 5. 逐行计算
# ==========================================
print("🧬 正在计算等电点差异和分子量差异...")

iso_diff_list = []
mw_diff_list = []
valid_count = 0

for idx, row in df.iterrows():
    wt, mt = parse_aa_change(row.get('Amino_acids', ''))
    if wt is not None and mt is not None:
        valid_count += 1

    iso_diff_list.append(calc_diff(wt, mt, AA_PI))
    mw_diff_list.append(calc_diff(wt, mt, AA_MW))

df['Isoelectric_diff'] = iso_diff_list
df['Molecular_weight'] = mw_diff_list

print(f"   有效氨基酸变化: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)")

# ==========================================
# 6. 导出
# ==========================================
df.to_csv(OUTPUT_CSV, index=False)
print("\n" + "=" * 70)
print("🏆 阶段 06 完美收官！氨基酸生化特征注入完毕！")
print(f"   新增特征: Isoelectric_diff, Molecular_weight")
print(f"💾 保存至: {OUTPUT_CSV}")
print("=" * 70)
