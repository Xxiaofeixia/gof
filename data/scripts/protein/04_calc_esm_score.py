import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import re
import glob
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import warnings
warnings.filterwarnings('ignore')

print("🚀 [Protein Pipeline 04] 正在唤醒 NVIDIA GPU 与 ESM-2 蛋白质大模型...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 当前计算核心: {device.upper()}")

# ==========================================
# 1. 加载 650M 超强参数 ESM-2
# ==========================================
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.eval() 

# ==========================================
# 2. 绝对路径配置与数据加载
# ==========================================
INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/03_BIOREASON_with_RSA.csv"
MAPPING_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/uniprot_mapping.csv"
PDB_DIR = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/alphafold_pdbs"
OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/04_BIOREASON_with_ESM.csv"

print("⏳ 加载带有 3D 特征的大表...")
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df_map = pd.read_csv(MAPPING_CSV)
    uniprot_dict = {str(k).strip(): str(v).strip() for k, v in zip(df_map['SYMBOL'], df_map['UniProt_ID'])}
except Exception as e:
    print(f"❌ 读取错误: {e}"); exit()

# ==========================================
# 3. 精细化清洗：严苛的氨基酸提取
# ==========================================
def extract_pos(x):
    match = re.search(r'\d+', str(x))
    return int(match.group()) if match else None

pos_col = 'Protein_position' if 'Protein_position' in df.columns else 'Amino_acids'
df['mut_pos'] = df[pos_col].apply(extract_pos)

def get_aa(x, index):
    aa_str = str(x).strip()
    # 🛡️ 核心安检：遇到终止密码子或格式不对，直接返回 None
    if pd.isna(x) or '*' in aa_str or aa_str == '-' or '/' not in aa_str:
        return None
    parts = aa_str.split('/')
    if len(parts) < 2: return None
    return parts[index].strip()[0].upper() if parts[index].strip() else None

df['WT_AA'] = df['Amino_acids'].apply(lambda x: get_aa(x, 0))
df['MUT_AA'] = df['Amino_acids'].apply(lambda x: get_aa(x, -1))

# ==========================================
# 4. 序列提取与大模型打分函数
# ==========================================
def get_seq_from_pdb(uid):
    search_pattern = os.path.join(PDB_DIR, f"AF-{uid}-F1-model_*.pdb")
    matched = glob.glob(search_pattern)
    if not matched: return None
    try:
        p = PDBParser(QUIET=True)
        struct = p.get_structure('X', matched[0])
        ppb = PPBuilder()
        seq = ""
        for pp in ppb.build_peptides(struct[0]):
            seq += str(pp.get_sequence())
        return seq
    except: return None

def calculate_esm_score(sequence, pos, wt_aa, mut_aa):
    # 🛡️ 如果前面的安检发现是无效氨基酸 (返回了 None)，这里直接给 0.0，不报错！
    if not sequence or pd.isna(pos) or pd.isna(wt_aa) or pd.isna(mut_aa): 
        return 0.0
    
    pos = int(pos)
    if pos < 1 or pos > len(sequence): return 0.0
    
    if sequence[pos-1] != str(wt_aa): 
        return 0.0 
    
    masked_seq = sequence[:pos-1] + tokenizer.mask_token + sequence[pos:]
    inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    wt_id = tokenizer.convert_tokens_to_ids(str(wt_aa))
    mut_id = tokenizer.convert_tokens_to_ids(str(mut_aa))
    
    score = logits[0, mask_idx, mut_id].item() - logits[0, mask_idx, wt_id].item()
    return round(score, 4)

# ==========================================
# 5. 执行 GPU 飞速推理
# ==========================================
print("🧬 开始向 GPU 喂数据... (已拦截无义突变，确保张量计算安全)")
esm_scores = []
for idx, row in df.iterrows():
    if idx > 0 and idx % 200 == 0: 
        print(f"   👉 GPU 已处理: {idx}/{len(df)}...")
        
    uid = uniprot_dict.get(str(row['SYMBOL']).strip())
    
    # 🛡️ 如果连突变位点都没有，直接跳过 PDB 解析，极速返回 0.0
    if pd.isna(row['WT_AA']) or pd.isna(row['MUT_AA']):
        esm_scores.append(0.0)
        continue
        
    seq = get_seq_from_pdb(uid)
    score = calculate_esm_score(seq, row['mut_pos'], row['WT_AA'], row['MUT_AA'])
    esm_scores.append(score)

df['ESM_DDG_Score'] = esm_scores
df.drop(columns=['mut_pos', 'WT_AA', 'MUT_AA'], inplace=True, errors='ignore')

df.to_csv(OUTPUT_CSV, index=False)
print("\n" + "="*70)
print("🏆 伟大的胜利！大模型结构稳定性 (ESM_DDG_Score) 全部注入完毕！")
print("="*70)