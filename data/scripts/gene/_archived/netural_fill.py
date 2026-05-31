import pandas as pd

print("🛡️ 启动【绝对同基因竞技场】清洗计划...")

# 1. 读取两个大表
df_data = pd.read_csv("BIOREASON_Final_Dataset_All_Features.csv", low_memory=False)
df_core = pd.read_csv("bioreason_final_result.csv")

# 2. 动态识别核心基因表的列名 (解决报错的核心！)
if 'Gene_Symbol' in df_core.columns:
    core_col = 'Gene_Symbol'
elif 'Gene' in df_core.columns:
    core_col = 'Gene'
else:
    # 如果都找不到，直接简单粗暴地拿第一列！
    core_col = df_core.columns[0] 
    print(f"⚠️ 提示：在核心基因表里没找到 SYMBOL 列，默认使用第一列 '{core_col}'。")

# 提取纯净的 574 基因白名单
core_genes = set(df_core[core_col].dropna().unique())
print(f"✅ 成功从护城河文件提取 {len(core_genes)} 个核心基因。")

# 3. 过滤大表 (VEP生成的大表里确定是有 SYMBOL 这一列的)
if 'SYMBOL' not in df_data.columns:
    print("❌ 致命错误：合并大表里没有 SYMBOL 列，请检查 VEP 输出是否正常！")
    exit()

df_strict = df_data[df_data['SYMBOL'].isin(core_genes)].copy()

# 4. 保存结果
df_strict.to_csv("BIOREASON_Final_Dataset_Strict.csv", index=False)

print("\n" + "="*40)
print(f"🏆 安检大杀青！最终剩下 {len(df_strict)} 条绝对纯血数据。")
print("⚖️ 最终天平分布：")
print(df_strict['LABEL'].value_counts())
print("="*40)