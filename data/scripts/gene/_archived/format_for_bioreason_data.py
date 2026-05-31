import pandas as pd

print("🚀 启动 BioReason 标准格式转化引擎...")

# 1. 加载上一步处理好的特征大表
INPUT_CSV = "BIOREASON_LLM_Ready.csv"
OUTPUT_CSV = "BioReason_Task_Dataset.csv"

try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
except Exception as e:
    print(f"❌ 找不到输入文件 {INPUT_CSV}，请检查路径！报错: {e}")
    exit()

print("⏳ 正在重构自然语言 Prompt (Question) 和 序列 (Sequences)...")

# 初始化 BioReason 格式的列表
bioreason_data = {
    "ID": [],
    "question": [],
    "answer": [],
    "reference_sequence": [],
    "variant_sequence": []
}

# 2. 逐行生成格式化数据
for idx, row in df.iterrows():
    # --- A. 生成 ID ---
    # 模仿截图里的格式 Task1_train_0, Task1_train_1 ...
    bioreason_data["ID"].append(f"Task_GOFLOF_train_{idx}")
    
    # --- B. 提取基础变量 ---
    gene = str(row.get('Gene', 'Unknown'))
    chrom = str(row['Location']).split(':')[0]
    
    # 提取特征 (处理空值)
    consequence = str(row.get('Consequence', 'unknown variant')).replace('_', ' ')
    aa_change = str(row.get('Amino_acids', 'unknown'))
    domains = str(row.get('DOMAINS', 'None'))
    max_af = row.get('MAX_AF', 0.0)
    am_score = row.get('AlphaMissense_score', 0.0)
    cadd = row.get('CADD_phred', 0.0)
    gerp = row.get('GERP++_RS', 0.0)
    label = str(row['LABEL'])
    
    # --- C. 构建大模型提示词 (Question) ---
    question = (
        f"The variant affects gene {gene} on Chromosome {chrom}. "
        f"The functional consequence is {consequence}, resulting in amino acid change {aa_change}. "
        f"It is located in the {domains} domain. "
        f"Its maximum population allele frequency is {max_af}. "
        f"Computational predictions indicate an AlphaMissense score of {am_score}, "
        f"a CADD score of {cadd}, and a GERP++ score of {gerp}. "
        f"Please evaluate whether this mutation is Gain-of-Function (GOF), Loss-of-Function (LOF), or Neutral."
    )
    bioreason_data["question"].append(question)
    
    # --- D. 构建标准答案 (Answer) ---
    if label == "GOF":
        ans = "Pathogenic; Gain-of-Function (GOF)"
    elif label == "LOF":
        ans = "Pathogenic; Loss-of-Function (LOF)"
    else:
        ans = "Benign; Neutral"
    bioreason_data["answer"].append(ans)
    
    # --- E. 序列处理 (Reference & Variant) ---
    # 核心修改：上一步已经生成了完美的双序列，这里直接读取赋值！
    ref_seq = str(row['reference_sequence'])
    var_seq = str(row['variant_sequence'])
    
    bioreason_data["reference_sequence"].append(ref_seq)
    bioreason_data["variant_sequence"].append(var_seq)

# 3. 生成最终的 DataFrame 并导出
df_bioreason = pd.DataFrame(bioreason_data)

df_bioreason.to_csv(OUTPUT_CSV, index=False)

print("="*50)
print(f"🏆 完美！已成功转换为 BioReason 官方训练格式！")
print(f"📊 数据预览 (共 {len(df_bioreason)} 条):")
print(df_bioreason[['ID', 'question', 'answer']].head(2))
print("="*50)