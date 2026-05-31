import pandas as pd
import numpy as np
import os, re, glob
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

print("🚀 [Protein Pipeline 05] 启动 3D 空间致病拓扑密度 (Spatial Density) 计算引擎...")

# ==========================================
# 1. 绝对路径配置
# ==========================================
INPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/04_BIOREASON_with_ESM.csv"
MAPPING_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/uniprot_mapping.csv"
PDB_DIR = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/alphafold_pdbs"
OUTPUT_CSV = "/gpfs/hpc/home/lijc/mapengtao/gof/data/processed/05_BIOREASON_with_Topology.csv"

# ==========================================
# 2. 数据加载
# ==========================================
print("⏳ 加载带有 ESM 打分的大表...")
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df_map = pd.read_csv(MAPPING_CSV)
    uniprot_dict = {str(k).strip(): str(v).strip() for k, v in zip(df_map['SYMBOL'], df_map['UniProt_ID'])}
except Exception as e:
    print(f"❌ 致命错误：读取失败，请确认 04 脚本已跑完。报错: {e}")
    exit()

# 提取并清洗突变位置 (现在咱们有了官方的 Protein_position，极其稳健！)
pos_col = 'Protein_position' if 'Protein_position' in df.columns else 'Amino_acids'
def extract_pos(x):
    match = re.search(r'\d+', str(x))
    return int(match.group()) if match else None

df['mut_pos'] = df[pos_col].apply(extract_pos)

# ==========================================
# 3. 按基因分组，计算 3D 空间距离
# ==========================================
density_dict = {}
grouped = df.groupby('SYMBOL')
total_genes = len(grouped)
current_gene = 0

print("🧬 开始构建 3D 坐标系，计算 10 埃 (Angstrom) 范围内的聚类密度...")
for symbol, group in grouped:
    current_gene += 1
    if current_gene % 50 == 0:
        print(f"   👉 正在处理第 {current_gene}/{total_genes} 个基因集群: {symbol}...")
        
    uid = uniprot_dict.get(str(symbol).strip())
    if not uid or uid == 'nan':
        for idx in group.index: density_dict[idx] = 0
        continue
        
    search_pattern = os.path.join(PDB_DIR, f"AF-{uid}-F1-model_*.pdb")
    matched = glob.glob(search_pattern)
    if not matched:
        for idx in group.index: density_dict[idx] = 0
        continue
        
    try:
        p = PDBParser(QUIET=True)
        structure = p.get_structure(uid, matched[0])
        model = structure[0]
            
        # 提取该蛋白质所有碳阿尔法 (CA) 原子的三维空间坐标 (X, Y, Z)
        ca_coords = {}
        for chain in model:
            for res in chain:
                res_num = res.get_id()[1]
                if 'CA' in res:
                    ca_coords[res_num] = res['CA'].get_coord()
        
        # 获取咱们表里该基因所有的突变位置
        mut_positions = group['mut_pos'].dropna().astype(int).unique()
        
        for idx, row in group.iterrows():
            pos = row['mut_pos']
            if pd.isna(pos):
                density_dict[idx] = 0
                continue
                
            pos = int(pos)
            density = 0
            if pos in ca_coords:
                pos_coord = ca_coords[pos]
                # 计算欧几里得距离，寻找 10 埃内的高危邻居
                for other_pos in mut_positions:
                    if other_pos != pos and other_pos in ca_coords:
                        dist = np.linalg.norm(pos_coord - ca_coords[other_pos])
                        if dist <= 10.0:
                            density += 1
            density_dict[idx] = density
            
    except Exception:
        # 容错：如果 PDB 文件损坏，当前基因密度记为 0
        for idx in group.index: density_dict[idx] = 0

# ==========================================
# 4. 挂载特征并保存
# ==========================================
df['Spatial_Density_10A'] = df.index.map(density_dict)
df.drop(columns=['mut_pos'], inplace=True, errors='ignore')

df.to_csv(OUTPUT_CSV, index=False)
print("\n" + "="*70)
print("🏆 阶段 05 完美收官！3D 空间拓扑密度特征注入完毕！")
print(f"💾 数据已妥善保存至: {OUTPUT_CSV}")
print("="*70)