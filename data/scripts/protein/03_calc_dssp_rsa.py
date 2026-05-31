import pandas as pd
import os
import re
import glob
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

print("🚀 [Protein Pipeline 03] 启动 DSSP 3D 结构物理特征鉴定引擎...")

# ==========================================
# 1. 路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "processed", "01_BIOREASON_with_DBs.csv")
MAPPING_CSV = os.path.join(BASE_DIR, "processed", "uniprot_mapping.csv")
PDB_DIR = os.path.join(BASE_DIR, "processed", "alphafold_pdbs")
OUTPUT_CSV = os.path.join(BASE_DIR, "processed", "03_BIOREASON_with_RSA.csv")

DSSP_BIN = "mkdssp"

# ==========================================
# 2. 加载大表与映射字典
# ==========================================
print("⏳ 正在加载基础数据与 UniProt 映射字典...")
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df_map = pd.read_csv(MAPPING_CSV)
    uniprot_dict = {str(k).strip(): str(v).strip() for k, v in zip(df_map['SYMBOL'], df_map['UniProt_ID'])}
except Exception as e:
    print(f"❌ 致命错误：读取 CSV 失败，请检查路径。报错: {e}")
    exit()

# ==========================================
# 3. 提取真实的氨基酸突变坐标与合法性校验
# ==========================================
pos_col = 'Protein_position' if 'Protein_position' in df.columns else 'Amino_acids'

def extract_pos(x):
    match = re.search(r'\d+', str(x))
    return int(match.group()) if match else None

# 🛡️ 核心安检：判断是不是标准的错义突变 (排除 *, -, 以及单字母)
def is_valid_missense(aa):
    aa_str = str(aa).strip()
    if pd.isna(aa) or '*' in aa_str or aa_str == '-' or '/' not in aa_str:
        return False
    return True

df['mut_pos'] = df[pos_col].apply(extract_pos)
df['is_missense'] = df['Amino_acids'].apply(is_valid_missense)

# ==========================================
# 4. 离线 DSSP 计算核心函数
# ==========================================
def calculate_3d_offline(uid, pos):
    if not uid or uid == 'nan' or pd.isna(pos): 
        return "Unknown", "Unknown"
    
    search_pattern = os.path.join(PDB_DIR, f"AF-{uid}-F1-model_*.pdb")
    matched_files = glob.glob(search_pattern)
    if not matched_files: return "Unknown", "Unknown"
    
    pdb_file = matched_files[0] 
    try:
        p = PDBParser(QUIET=True)
        structure = p.get_structure(uid, pdb_file)
        dssp = DSSP(structure[0], pdb_file, dssp=DSSP_BIN) 
        
        for key in dssp.keys():
            res_num = key[1][1] if isinstance(key[1], tuple) else key[1]
            if res_num == pos:
                return dssp[key][2], dssp[key][3] 
    except Exception:
        pass 
    return "Unknown", "Unknown"

# ==========================================
# 5. 遍历计算
# ==========================================
print("🧬 开始调用 DSSP... (已开启智能跳过非错义突变功能)")
ss_list, rsa_list = [], []

for idx, row in df.iterrows():
    if idx > 0 and idx % 200 == 0: 
        print(f"   👉 进度: {idx}/{len(df)}...")
    
    # 🛡️ 如果不是错义突变，直接赋予 Unknown，不浪费算力去读取 PDB
    if not row['is_missense']:
        ss_list.append("Unknown")
        rsa_list.append("Unknown")
        continue

    symbol = str(row['SYMBOL']).strip()
    uid = uniprot_dict.get(symbol)
    
    ss, rsa = calculate_3d_offline(uid, row['mut_pos'])
    ss_list.append(ss)
    rsa_list.append(rsa)

# ==========================================
# 6. 保存特征
# ==========================================
df['Secondary_Structure'] = ss_list
df['AlphaFold_RSA'] = rsa_list
# 清理临时列
df.drop(columns=['mut_pos', 'is_missense'], inplace=True, errors='ignore') 

df.to_csv(OUTPUT_CSV, index=False)
print("\n" + "="*70)
print("🏆 阶段 03 完美收官！3D 结构特征 (SS, RSA) 已成功注入！")
print("="*70)