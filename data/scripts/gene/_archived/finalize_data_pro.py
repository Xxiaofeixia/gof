import pandas as pd
import pysam
import numpy as np

print("🚀 启动大模型终极特征提纯 & DNA 双序列提取引擎 (全模态特征版)...")
# --- 1. 核心路径配置 ---
# 【关键修改】：输入文件必须是我们刚才跑完空间密度和 ESM 打分的终极大表
INPUT_CSV = "./source/BIOREASON_Ultimate_Topology.csv" 
OUTPUT_CSV = "./source/BIOREASON_LLM_Ready.csv"
FASTA_PATH = "/gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta"
FLANKING_BP = 2000 # 左右各 2000bp

# --- 2. 锁定大模型专属“王牌特征” ---
KEPT_COLUMNS = [
    # 1. 绝对标签与追踪
    'LABEL', 'Location', 'Allele', 'Gene', 'SYMBOL',  
    # 2. 语义与结构定位
    'Consequence', 'DOMAINS', 'Amino_acids', 
    # 3. 人群频率 (照妖镜)
    'AF', 'MAX_AF', 'gnomADe_AF',             
    # 4. 深度学习与进化保守性
    'AlphaMissense_score', 'CADD_phred',      
    'GERP++_RS', 'phyloP100way_vertebrate',
    # 5. 【新增】宏观遗传模式与基因必需性
    'MOI_pred', 'Inheritance_Pattern', 
    'Haploinsufficiency_Score', 'Gene_Essentiality',
    # 6. 【新增】3D 空间拓扑与大模型热力学
    'Secondary_Structure', 'AlphaFold_RSA', 'Spatial_Density_10A',
    'ESM_DDG_Score', 'Predicted_Stability_Change'
]

# --- 3. 加载与提纯 ---
print("⏳ 正在加载包含 3D 与宏观特征的原始大表...")
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
except Exception as e:
    print(f"❌ 致命错误：找不到合并后的大表 {INPUT_CSV}，请检查路径！")
    exit()

# 动态匹配列名（防止由于版本不同导致的列名微小差异）
actual_cols = [col for col in KEPT_COLUMNS if col in df.columns]
df = df[actual_cols].copy()

# --- 4. 空值清洗 (防止模型作弊) ---
print("🛡️ 正在清理特征空值，执行数值化与文本转化...")
df.replace('-', np.nan, inplace=True)

# A. 数值型特征：强制转为 float，空值填 0
score_cols = ['AlphaMissense_score', 'CADD_phred', 'GERP++_RS', 'phyloP100way_vertebrate', 'AF', 'MAX_AF', 'gnomADe_AF', 'Spatial_Density_10A']
for col in score_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# B. 文本/混合型特征：空值或解析失败填 "Unknown" 或 "None" (防止强制转 0 误导模型)
text_cols = ['Consequence', 'DOMAINS', 'Amino_acids', 'MOI_pred', 'Inheritance_Pattern', 
             'Haploinsufficiency_Score', 'Gene_Essentiality', 'Secondary_Structure', 
             'AlphaFold_RSA', 'ESM_DDG_Score', 'Predicted_Stability_Change']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# --- 5. 挂载基因组，提取双序列 (Reference & Variant) ---
print("✂️ 正在挂载参考基因组，提取变异前后的 DNA 双序列...")
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
        chrom_part = loc.split(':')[0]
        pos_part = loc.split(':')[1].split('-')[0]
        
        chrom = chrom_part if chrom_part.startswith('chr') else f"chr{chrom_part}"
        pos_1based = int(pos_part)
        
        start_0based = max(0, pos_1based - FLANKING_BP - 1)
        end_0based = pos_1based + FLANKING_BP
        
        # 1. 提取参考序列 (变异前)
        ref_seq = fasta.fetch(reference=chrom, start=start_0based, end=end_0based).upper()
        
        # 2. 构造变异序列 (变异后)
        alt_allele = str(row['Allele']).upper()
        # 针对 SNV 进行中心碱基替换
        var_seq = ref_seq[:FLANKING_BP] + alt_allele + ref_seq[FLANKING_BP + 1:]
        
        ref_sequences.append(ref_seq)
        var_sequences.append(var_seq)
    except Exception as e:
        ref_sequences.append("ERROR")
        var_sequences.append("ERROR")
        failed_count += 1

# 直接命名为框架要求的前后双序列列名
df['reference_sequence'] = ref_sequences
df['variant_sequence'] = var_sequences

# 丢弃极少数切取失败的异常坐标
df_final = df[(df['reference_sequence'] != "ERROR") & (df['variant_sequence'] != "ERROR")].copy()

# --- 6. 导出完美结果 ---
df_final.to_csv(OUTPUT_CSV, index=False)

print("="*60)
print(f"🏆 杀青！最终大模型多模态训练集（含全部特征 & 序列）已生成！")
print(f"📊 有效数据共 {len(df_final)} 条，序列提取失败 {failed_count} 条。")
print(f"💾 保存在: {OUTPUT_CSV}")
print("="*60)
