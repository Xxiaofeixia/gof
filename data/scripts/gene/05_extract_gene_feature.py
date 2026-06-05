import pandas as pd
import pysam
import numpy as np
import io
import os

print("🚀 启动大模型终极特征提纯 & DNA 双序列提取引擎...")

# --- 1. 核心路径配置 (全部使用你的绝对路径) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEP_TXT_PATH = os.path.join(BASE_DIR, "processed", "master_vep_output.txt")
RAW_LABELS_CSV = os.path.join(BASE_DIR, "processed", "master_raw_all_labels.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "processed", "BIOREASON_with_all_gene.csv")
FASTA_PATH = "/gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta"
FLANKING_BP = 2000 # 左右各 2000bp

# --- 2. 智能读取 VEP TXT ---
print("⏳ 正在智能解析 VEP 注释文本...")
with open(VEP_TXT_PATH, 'r') as f:
    # 过滤掉 VEP 开头所有的双井号 ## 注释行，只留表头和数据
    lines = [line for line in f if not line.startswith('##')]

# 使用 pandas 读取 Tab 分隔的数据
df_vep = pd.read_csv(io.StringIO(''.join(lines)), sep='\t', low_memory=False)
# 把奇怪的表头清理一下
df_vep.rename(columns={'#Uploaded_variation': 'ID'}, inplace=True)

# 🚨 去重：同一个变异会被 VEP 注释出多个转录本，按"影响力"排序后保留最佳行
# 排序优先级：1) 有氨基酸变化 > 没有  2) CANONICAL 转录本优先  3) 后果严重程度从高到低
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
    sort_keys.append((conseq_rank, is_canon, -has_aa))  # has_aa=1排前面 → 用负数

df_vep['_sort_key'] = sort_keys
df_vep = df_vep.sort_values('_sort_key').drop_duplicates(subset=['ID'], keep='first').copy()
df_vep.drop(columns=['_sort_key'], inplace=True, errors='ignore')

# --- 3. 缝合绝对标签 ---
print("🔗 正在缝合 GOF/LOF/Neutral 绝对标签...")
df_labels = pd.read_csv(RAW_LABELS_CSV)
# 通过变异的身份证(ID)将 VEP 特征和原始标签完美拼合
df = pd.merge(df_vep, df_labels[['ID', 'LABEL', 'REF', 'GENE_TYPE']], on='ID', how='inner')

# --- 4. 锁定大模型专属“王牌特征” ---
# 有些列如果在 VEP 里没跑出来，咱们也不报错，智能跳过
TARGET_COLS = [
    'ID', 'LABEL', 'Location', 'Allele', 'Gene', 'SYMBOL', 
    'Consequence', 'DOMAINS', 'Amino_acids','Protein_position',
    'AF', 'MAX_AF', 'AlphaMissense_score', 'CADD_phred', 
    'GERP++_RS', 'phyloP100way_vertebrate'
]
actual_cols = [col for col in TARGET_COLS if col in df.columns]
extra_cols = ['REF'] if 'REF' not in actual_cols else []
if 'GENE_TYPE' in df.columns:
    extra_cols.append('GENE_TYPE')
df = df[actual_cols + extra_cols].copy()

# --- 5. 空值清洗 (防止模型作弊) ---
print("🛡️ 正在清理特征空值，执行数值化转换...")
df.replace('-', np.nan, inplace=True)

score_cols = ['AlphaMissense_score', 'CADD_phred', 'GERP++_RS', 'phyloP100way_vertebrate', 'AF', 'MAX_AF']
for col in score_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

text_cols = ['Consequence', 'DOMAINS', 'Amino_acids', 'SYMBOL']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("None")

# --- 6. 挂载基因组，提取双序列 (Reference & Variant) ---
print(f"✂️ 正在挂载参考基因组 (截取上下游 {FLANKING_BP}bp)...")
try:
    fasta = pysam.FastaFile(FASTA_PATH)
except Exception as e:
    print(f"❌ 致命错误：无法加载 Fasta 文件。请检查路径或索引！\n报错: {e}")
    exit()

ref_sequences = []
var_sequences = []
failed_count = 0

for index, row in df.iterrows():
    try:
        loc = str(row['Location'])
        chrom_part, pos_part = loc.split(':')
        pos_part = pos_part.split('-')[0]
        
        chrom = chrom_part if chrom_part.startswith('chr') else f"chr{chrom_part}"
        pos_1based = int(pos_part)
        
        start_0based = max(0, pos_1based - FLANKING_BP - 1)
        # 这里严谨起见，考虑参考碱基的长度
        ref_len = len(str(row['REF'])) if pd.notna(row['REF']) else 1
        end_0based = pos_1based + FLANKING_BP + (ref_len - 1)
        
        # 1. 提取参考序列 (变异前)
        ref_seq = fasta.fetch(reference=chrom, start=start_0based, end=end_0based).upper()
        
        # 2. 构造变异序列 (变异后)
        alt_allele = str(row['Allele']).upper()
        var_seq = ref_seq[:FLANKING_BP] + alt_allele + ref_seq[FLANKING_BP + ref_len:]
        
        # 截断回标准长度，保证输入大模型的 token 数量绝对一致
        ref_seq = ref_seq[:FLANKING_BP*2 + 1]
        var_seq = var_seq[:FLANKING_BP*2 + 1]
        
        ref_sequences.append(ref_seq)
        var_sequences.append(var_seq)
    except Exception as e:
        ref_sequences.append("ERROR")
        var_sequences.append("ERROR")
        failed_count += 1

df['reference_sequence'] = ref_sequences
df['variant_sequence'] = var_sequences

# 丢弃提取失败的异常行
df_final = df[(df['reference_sequence'] != "ERROR") & (df['variant_sequence'] != "ERROR")].copy()

# 整理一下列顺序，让表格更美观
# 把漏掉的黄金列全部加回来！
cols_order = [
    'ID', 'LABEL', 'SYMBOL', 'Gene', 'Location', 'Allele', 
    'Consequence', 'Amino_acids', 'DOMAINS', 'Protein_position',
] + [c for c in score_cols if c in df_final.columns] + [
    'reference_sequence', 'variant_sequence'
]

df_final = df_final[[c for c in cols_order if c in df_final.columns]]

# --- 7. 导出完美结果 ---
df_final.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*50)
print(f"🏆 杀青！最终大模型多模态训练集已生成！")
print(f"📊 有效数据共 {len(df_final)} 条，提取失败 {failed_count} 条。")
print(f"💾 保存在: {OUTPUT_CSV}")
print("="*50)
