import pandas as pd
import requests
import os
import time
import urllib3

# 禁用不安全的 HTTPS 警告，防止日志被刷屏
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("🚀 [Protein Pipeline 02] 启动 UniProt 映射与 AlphaFold 3D 结构精确下载器...")

# ==========================================
# 1. 路径配置 (严谨对齐 01 的输出)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "processed", "01_BIOREASON_with_DBs.csv")

# 字典和结构文件夹直接放在 processed 下，供后续脚本调用
MAPPING_CSV = os.path.join(BASE_DIR, "processed", "uniprot_mapping.csv")
PDB_DIR = os.path.join(BASE_DIR, "processed", "alphafold_pdbs")
os.makedirs(PDB_DIR, exist_ok=True)

# ==========================================
# 2. 读取数据，提取独立基因
# ==========================================
try:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    unique_genes = df['SYMBOL'].dropna().unique()
    print(f"🔍 成功读取 01 号大表，共发现 {len(unique_genes)} 个需要匹配 3D 结构的独立基因。")
except Exception as e:
    print(f"❌ 致命错误：读取 {INPUT_CSV} 失败，请确认 01 脚本是否成功跑完。报错: {e}")
    exit()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# ==========================================
# 3. 阶段 A：获取 UniProt ID (强制官方审核版)
# ==========================================
uniprot_mapping = {}
print("\n🌐 [阶段 A] 正在查询 UniProt ID (强制匹配 reviewed:true 官方经典蛋白)...")

for i, gene in enumerate(unique_genes):
    gene = str(gene).strip()
    if i > 0 and i % 50 == 0: 
        print(f"   👉 已查询 {i}/{len(unique_genes)} 个基因...")
        
    try:
        # 核心逻辑：只查人类(9606) 且 经过人工审核(reviewed:true) 的经典主序列
        url = f"https://rest.uniprot.org/uniprotkb/search?query=(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:true)&fields=accession&size=1"
        res = requests.get(url, timeout=15, verify=False) 
        if res.status_code == 200:
            data = res.json()
            if data.get('results'):
                uniprot_mapping[gene] = data['results'][0]['primaryAccession']
    except Exception:
        pass # 容错处理：网络波动不中断，直接跳过
    time.sleep(0.1) # 礼貌延迟，防止被 UniProt 封 IP

# 保存映射字典，不仅为了本次，更为了以后断点续传和 DSSP 调用
df_map = pd.DataFrame(list(uniprot_mapping.items()), columns=['SYMBOL', 'UniProt_ID'])
df_map.to_csv(MAPPING_CSV, index=False)
print(f"✅ 映射完成！成功获取了 {len(df_map)} 个经典 UniProt ID，已保存至 {MAPPING_CSV}。")

# ==========================================
# 4. 阶段 B：官方 API 精确获取 PDB 文件
# ==========================================
print(f"\n📥 [阶段 B] 开始通过 AlphaFold 官方 API 精确下载 PDB 文件...")
success_count = 0

for i, uid_raw in enumerate(df_map['UniProt_ID'].dropna()):
    uid = str(uid_raw).strip()
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
    
    try:
        # 第一步：向官方 API 问路，获取真实下载链接 (不再盲猜 v4 还是 v3)
        res = requests.get(api_url, headers=headers, timeout=15, verify=False)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                exact_pdb_url = data[0].get('pdbUrl') 
                
                if exact_pdb_url:
                    file_name = exact_pdb_url.split('/')[-1]
                    save_path = os.path.join(PDB_DIR, file_name)
                    
                    # 第二步：检查本地是否已存在（完美支持断点续传）
                    if not os.path.exists(save_path):
                        r = requests.get(exact_pdb_url, headers=headers, timeout=30, verify=False)
                        if r.status_code == 200:
                            with open(save_path, 'wb') as f:
                                f.write(r.content)
                            success_count += 1
                        else:
                            print(f"⚠️ {uid}: 下载链接有效但文件请求失败 (HTTP {r.status_code})")
                    else:
                        success_count += 1 # 文件已存在，算作成功
                else:
                    print(f"⚠️ {uid}: 官方有记录，但未提供 PDB 格式链接")
            else:
                print(f"⚠️ {uid}: AlphaFold 数据库尚未收录")
        elif res.status_code == 404:
            print(f"⚠️ {uid}: AlphaFold 查无此蛋白 (HTTP 404)")
        else:
            print(f"⚠️ {uid}: API 拒绝查询 (HTTP {res.status_code})")
    except Exception as e:
        print(f"❌ {uid}: 处理发生网络异常 -> {e}")
        
    time.sleep(0.2)
    
    if i > 0 and i % 50 == 0: 
        print(f"   👉 PDB 下载进度: {i}/{len(df_map)}...")

print("\n" + "="*70)
print(f"🎉 阶段 02 完美收官！共成功锁定并准备好了 {success_count} 个 3D PDB 结构文件！")
print(f"📁 结构库保存在: {PDB_DIR}/")
print("="*70)
