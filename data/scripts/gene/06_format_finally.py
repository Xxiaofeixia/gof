import pandas as pd
import os

print("🚀 启动大模型标准训练集格式转化引擎 (BioReason Format)...")

# --- 1. 绝对路径配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "processed", "BIOREASON_with_all_gene.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "processed", "BioReason_gene_Dataset_LLM_Ready.csv")

# --- 2. 加载数据 ---
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
except Exception as e:
    print(f"❌ 找不到输入文件 {INPUT_CSV}，请检查是否跑完了上一步提取脚本！\n报错: {e}")
    exit()

print(f"⏳ 正在为 {len(df)} 条数据重构自然语言 Prompt (Question) 和双序列...")

bioreason_data = {
    "ID": [],
    "question": [],
    "answer": [],
    "reference_sequence": [],
    "variant_sequence": []
}

# --- 3. 逐行生成格式化指令数据 ---
for idx, row in df.iterrows():
    # --- A. 生成溯源 ID ---
    var_id = str(row.get('ID', f"var_{idx}"))
    bioreason_data["ID"].append(f"Task_GOFLOF_{var_id}")
    
    # --- B. 提取基础变量 (🌟 核心修改：使用 SYMBOL 替代难懂的 Ensembl ID) ---
    symbol = str(row.get('SYMBOL', 'Unknown_Gene'))
    loc = str(row.get('Location', 'Unknown_Location'))
    chrom = loc.split(':')[0] if ':' in loc else 'Unknown_Chr'
    
    # 提取文本特征并清理下划线，让语言更自然
    consequence = str(row.get('Consequence', 'unknown variant')).replace('_', ' ')
    aa_raw = str(row.get('Amino_acids', ''))
    aa_change = aa_raw if aa_raw not in ('', '-', 'None', 'nan', 'unknown') else None
    domains_raw = str(row.get('DOMAINS', ''))
    domains = domains_raw if domains_raw not in ('', '-', 'None', 'nan') else 'unknown'

    # 提取数值打分
    max_af = row.get('MAX_AF', 0.0)
    am_score = row.get('AlphaMissense_score', 0.0)
    cadd = row.get('CADD_phred', 0.0)
    gerp = row.get('GERP++_RS', 0.0)
    phylop = row.get('phyloP100way_vertebrate', 0.0)
    label = str(row.get('LABEL', 'Unknown')).upper()

    # --- C. 🧬 构建大模型提示词 (Question)，氨基酸变化仅在已知时写入 ---
    aa_part = f", resulting in amino acid change {aa_change}" if aa_change else ""
    question = (
        f"The variant affects gene {symbol} on Chromosome {chrom} at location {loc}. "
        f"The functional consequence is {consequence}{aa_part}. "
        f"It is located in the {domains} domain. "
        f"Its maximum population allele frequency is {max_af}. "
        f"Computational predictions indicate an AlphaMissense score of {am_score}, "
        f"a CADD score of {cadd}, a GERP++ score of {gerp}, and a phyloP100way score of {phylop}. "
        f"Based on these features and the provided DNA sequences, please evaluate whether this mutation is "
        f"Gain-of-Function (GOF), Loss-of-Function (LOF), or Neutral."
    )
    bioreason_data["question"].append(question)
    
    # --- D. 🎯 构建标准答案 (Answer) ---
    if label == "GOF":
        ans = "Pathogenic; Gain-of-Function (GOF)"
    elif label == "LOF":
        ans = "Pathogenic; Loss-of-Function (LOF)"
    elif label == "NEUTRAL":
        ans = "Benign; Neutral"
    else:
        ans = f"Unknown; {label}"
    bioreason_data["answer"].append(ans)
    
    # --- E. 🧬 序列处理 (直接继承上一步的完美双序列) ---
    bioreason_data["reference_sequence"].append(str(row.get('reference_sequence', '')))
    bioreason_data["variant_sequence"].append(str(row.get('variant_sequence', '')))

# --- 4. 生成 DataFrame 并导出 ---
df_bioreason = pd.DataFrame(bioreason_data)
df_bioreason.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*50)
print(f"🏆 完美收官！已成功转换为大模型官方微调格式 (BioReason Format)！")
print(f"📊 最终可用微调数据: {len(df_bioreason)} 条。")
print(f"💾 数据已妥善保存至: {OUTPUT_CSV}")
print("="*50)
