import pandas as pd
import os

print("🚀 [Protein Pipeline 01] 启动宏观数据库(gnomAD + ClinGen)融合引擎...")

# ==========================================
# 1. 路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 输入：基因流水线产物 (直接读取，跳过中间改名步骤)
INPUT_BASE_CSV = os.path.join(BASE_DIR, "processed", "BIOREASON_with_all_gene.csv")

# 输入：原始数据库文件 (已转移至 raw 目录)
GNOMAD_FILE = os.path.join(BASE_DIR, "raw", "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz")
CLINGEN_FILE = os.path.join(BASE_DIR, "raw", "MOIExport.csv")

# 输出：阶段 01 的产物
OUTPUT_CSV = os.path.join(BASE_DIR, "processed", "01_BIOREASON_with_DBs.csv")

# ==========================================
# 2. 加载基础数据
# ==========================================
print(f"⏳ 正在加载基因阶段基座大表...")
try:
    df_main = pd.read_csv(INPUT_BASE_CSV, low_memory=False)
    print(f"✅ 基座表加载成功，当前数据量: {len(df_main)} 条。")
except Exception as e:
    print(f"❌ 致命错误：找不到输入文件！请检查路径是否正确。\n报错详情: {e}")
    exit()

# ==========================================
# 3. 融合 gnomAD 约束特征
# ==========================================
print("⏳ 正在解析并融合 gnomAD 基因约束大表...")
try:
    # pandas 可以直接读取 bgz/gz 压缩包，极其优雅
    df_gnomad = pd.read_csv(GNOMAD_FILE, sep="\t", compression='gzip', low_memory=False)
    df_gnomad_sub = df_gnomad[['gene', 'pLI', 'oe_lof_upper']].copy()
    df_gnomad_sub.rename(columns={'gene': 'SYMBOL', 'oe_lof_upper': 'Haploinsufficiency_Score'}, inplace=True)
    
    # 将 gnomAD 数据级联入主表
    df_main = pd.merge(df_main, df_gnomad_sub, on='SYMBOL', how='left')
    
    # 填充缺失值：如果某个基因在 gnomAD 没有记录，不能填 0.0，填 Unknown 最严谨
    df_main['pLI'] = df_main['pLI'].fillna("Unknown")
    df_main['Haploinsufficiency_Score'] = df_main['Haploinsufficiency_Score'].fillna("Unknown")
    
    print("✅ gnomAD 核心特征 (pLI, oe_lof_upper) 注入成功！")
except Exception as e:
    print(f"❌ 致命错误：gnomAD 融合失败！请检查 raw 目录下是否有该压缩包。\n报错详情: {e}")
    exit()

# ==========================================
# 4. 融合 ClinGen MOI (遗传模式)
# ==========================================
print("⏳ 正在解析并融合 ClinGen 遗传模式数据库...")
try:
    df_clingen = pd.read_csv(CLINGEN_FILE)
    
    # 核心清洗：剔除恼人的 HGNC 编号后缀 (如 "A2ML1HGNC:23336")，只保留纯基因名
    df_clingen['SYMBOL'] = df_clingen['Gene'].apply(lambda x: str(x).split('HGNC')[0].strip())
    df_clingen_sub = df_clingen[['SYMBOL', 'MOI']].dropna(subset=['MOI']).copy()
    
    # 智能去重合并：同一个基因如果有多个 MOI (如 AD, AR)，合并为 "AD|AR"
    df_clingen_unique = df_clingen_sub.groupby('SYMBOL')['MOI'].apply(
        lambda x: '|'.join(x.unique())
    ).reset_index()
    df_clingen_unique.rename(columns={'MOI': 'Inheritance_Pattern'}, inplace=True)
    
    # 级联到主表
    df_main = pd.merge(df_main, df_clingen_unique, on='SYMBOL', how='left')
    df_main['Inheritance_Pattern'] = df_main['Inheritance_Pattern'].fillna("Unknown")
    
    print("✅ ClinGen 宏观遗传特征 (Inheritance_Pattern) 注入成功！")
except Exception as e:
    print(f"❌ 致命错误：ClinGen 融合失败！请检查 raw 目录下是否有 MOIExport.csv。\n报错详情: {e}")
    exit()

# ==========================================
# 5. 导出本阶段成果
# ==========================================
df_main.to_csv(OUTPUT_CSV, index=False)
print("\n" + "="*70)
print(f"🏆 阶段 01 完美收官！带有宏观数据库约束的新表已生成！")
print(f"💾 数据已妥善保存至: {OUTPUT_CSV}")
print("="*70)
