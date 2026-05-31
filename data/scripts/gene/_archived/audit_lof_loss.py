import pandas as pd

print("🕵️‍♂️ 启动 LOF 数据流失专项审计...\n")

# 1. 你升维前/提取时的原始 LOF 弹药库 (请确认你的文件名)
raw_lof_file = "lof_all_types_to_annotate.vcf" 
# 2. 你的 574 核心基因白名单
core_genes_file = "bioreason_final_result.csv"
# 3. 你最终清洗完毕的极严苛大表
final_strict_file = "BIOREASON_LLM_Ready.csv"

# --- 环节 A：查看原始弹药库 ---
try:
    # 读 VCF，跳过 # 开头的注释行
    with open(raw_lof_file, 'r') as f:
        raw_count = sum(1 for line in f if not line.startswith('#'))
    print(f"▶️ 1. 原始提炼的 LOF (hg19时期): {raw_count} 条")
except:
    print("❌ 找不到原始 lof vcf 文件")

# --- 环节 B：查看最终幸存者 ---
try:
    df_final = pd.read_csv(final_strict_file, low_memory=False)
    # 过滤出最终的 LOF
    df_final_lof = df_final[df_final['LABEL'] == 'LOF']
    final_count = len(df_final_lof)
    print(f"▶️ 2. 最终进入大模型的纯血 LOF: {final_count} 条")
    print(f"🔻 总体真实折损: 消失了 {raw_count - final_count} 条！")
    
    # --- 环节 C：揪出真凶（统计不在 574 名单里的基因） ---
    df_core = pd.read_csv(core_genes_file)
    gene_col = 'SYMBOL' if 'SYMBOL' in df_core.columns else df_core.columns[0]
    core_genes_set = set(df_core[gene_col].dropna().unique())
    
    # 假设你的大表里有基因列
    if 'SYMBOL' in df_final.columns:
        print(f"\n💡 进一步分析幸存的 {final_count} 条 LOF，它们分布在 {df_final_lof['SYMBOL'].nunique()} 个核心基因上。")
        print("  也就是说，你虽然有 574 个核心基因，但你提取的这段 LOF 数据，并没有覆盖所有的基因。")

except Exception as e:
    print(f"❌ 统计失败: {e}")
