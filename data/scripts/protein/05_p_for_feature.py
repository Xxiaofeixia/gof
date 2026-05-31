import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🔬 启动多模态特征显著性 (p-value) 统计验证引擎...")

# ==========================================
# 1. 配置与数据加载
# ==========================================
INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/05_BIOREASON_with_Topology.csv"
OUTPUT_PVAL_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/Feature_P_Values_Report.csv"

try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"✅ 成功加载数据，共 {len(df)} 条记录。")
except Exception as e:
    print(f"❌ 读取数据失败: {e}")
    exit()

# 确保标签规范化
df['LABEL'] = df['LABEL'].str.strip().str.upper()

# ==========================================
# 2. 特征预处理 (极其关键：清理 Unknown)
# ==========================================
# 定义需要验证的连续型数值特征
num_features = [
    'MAX_AF', 'phyloP100way_vertebrate', 'GERP++_RS', 
    'AlphaMissense_score', 'CADD_phred', 'Haploinsufficiency_Score', 
    'AlphaFold_RSA', 'Spatial_Density_10A', 'ESM_DDG_Score'
]

# 定义离散型分类特征
cat_features = ['Secondary_Structure', 'Has_Domain']

# 处理结构域 (二值化：有结构域 vs 无结构域)
df['Has_Domain'] = df['DOMAINS'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() in ('', '-', 'None', 'nan') else 1)

# 将数值特征中的干扰项 (如 '-' 或 'Unknown') 替换为 NaN，并转为浮点数
for col in num_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].replace(['Unknown', '-', 'None', 'nan'], np.nan), errors='coerce')

# 二级结构中的 Unknown 也要设为 NaN
if 'Secondary_Structure' in df.columns:
    df['Secondary_Structure'] = df['Secondary_Structure'].replace(['Unknown', '-', 'None', 'nan'], np.nan)

# ==========================================
# 3. 统计学引擎 (双阶段检测)
# ==========================================
def calculate_p_values(df_subset, group_col, group_A, group_B):
    """计算两组之间的特征 p 值"""
    results = []
    
    # 过滤出只有 A 和 B 组的数据
    temp_df = df_subset[df_subset[group_col].isin([group_A, group_B])]
    
    # A. 数值特征 (Mann-Whitney U 秩和检验)
    for feat in num_features:
        if feat not in temp_df.columns: continue
        
        data_A = temp_df[temp_df[group_col] == group_A][feat].dropna()
        data_B = temp_df[temp_df[group_col] == group_B][feat].dropna()
        
        if len(data_A) > 10 and len(data_B) > 10:
            stat, p_val = stats.mannwhitneyu(data_A, data_B, alternative='two-sided')
            results.append({'Feature': feat, 'Type': 'Numerical', 'p_value': p_val})
        else:
            results.append({'Feature': feat, 'Type': 'Numerical', 'p_value': np.nan})
            
    # B. 分类特征 (卡方检验)
    for feat in cat_features:
        if feat not in temp_df.columns: continue
        
        # 构建交叉表 (剔除 NaN)
        cross_tab = pd.crosstab(temp_df[temp_df[feat].notna()][group_col], 
                                temp_df[temp_df[feat].notna()][feat])
        
        if cross_tab.shape[0] == 2 and cross_tab.shape[1] > 1:
            stat, p_val, dof, expected = stats.chi2_contingency(cross_tab)
            results.append({'Feature': feat, 'Type': 'Categorical', 'p_value': p_val})
        else:
            results.append({'Feature': feat, 'Type': 'Categorical', 'p_value': np.nan})
            
    return pd.DataFrame(results)

# ---------------------------------------------------------
# 阶段一：Pathogenic (GOF+LOF) vs. Benign (Neutral)
# ---------------------------------------------------------
print("📊 正在执行 [阶段一] 计算：Pathogenic vs Neutral ...")
df['Stage1_Label'] = df['LABEL'].apply(lambda x: 'Pathogenic' if x in ['GOF', 'LOF'] else ('Neutral' if x == 'NEUTRAL' else np.nan))
df_stage1 = calculate_p_values(df, 'Stage1_Label', 'Pathogenic', 'Neutral')
df_stage1.rename(columns={'p_value': 'Stage1_p_value (Pathogenic vs Neutral)'}, inplace=True)

# ---------------------------------------------------------
# 阶段二：GOF vs. LOF
# ---------------------------------------------------------
print("📊 正在执行 [阶段二] 计算：GOF vs LOF ...")
df_stage2 = calculate_p_values(df, 'LABEL', 'GOF', 'LOF')
df_stage2.rename(columns={'p_value': 'Stage2_p_value (GOF vs LOF)'}, inplace=True)

# ==========================================
# 4. 报表合并与格式化
# ==========================================
# 合并两个阶段的报告
final_report = pd.merge(df_stage1, df_stage2[['Feature', 'Stage2_p_value (GOF vs LOF)']], on='Feature', how='outer')

# 科学计数法格式化 (方便阅读，比如 1.2e-5)
final_report['Stage1_p_value (Pathogenic vs Neutral)'] = final_report['Stage1_p_value (Pathogenic vs Neutral)'].apply(lambda x: f"{x:.4e}" if pd.notna(x) else "N/A")
final_report['Stage2_p_value (GOF vs LOF)'] = final_report['Stage2_p_value (GOF vs LOF)'].apply(lambda x: f"{x:.4e}" if pd.notna(x) else "N/A")

# 按照阶段二 (区分 GOF/LOF 最难的阶段) 的 p 值排序
# 临时转回 float 排序后再转回来
final_report['_sort_val'] = pd.to_numeric(final_report['Stage2_p_value (GOF vs LOF)'], errors='coerce').fillna(1)
final_report = final_report.sort_values('_sort_val').drop(columns=['_sort_val'])

# 导出 CSV
final_report.to_csv(OUTPUT_PVAL_CSV, index=False)

print("\n" + "="*70)
print(f"🎉 特征 p 值验证完成！")
print(f"💾 详细报表已保存至: {OUTPUT_PVAL_CSV}")
print("="*70)
print("\n👇 极度显著的特征预览 (Top 5 区分 GOF/LOF):")
print(final_report.head(5).to_string(index=False))
