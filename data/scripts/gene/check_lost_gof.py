import pandas as pd
from pyliftover import LiftOver
import os

print("🕵️‍♂️ 启动【gofcards 变异损耗法医鉴定程序】...")

# ================= 1. 路径配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAIN_FILE  = os.path.join(BASE_DIR, “reference”, “hg19ToHg38.over.chain.gz”)
GOF_EXCEL   = os.path.join(BASE_DIR, “raw”, “gofcards_data_download.xlsx”)
# 输出一份详细的”阵亡名单”供你复核
FAILED_LOG_CSV = os.path.join(BASE_DIR, “processed”, “gofcards_attrition_log.csv”)

# ================= 2. 检查文件与加载 LiftOver =================
try:
    lo = LiftOver(CHAIN_FILE)
    df = pd.read_excel(GOF_EXCEL)
except Exception as e:
    print(f"❌ 文件加载失败，请检查路径。报错: {e}")
    exit()

total_rows = len(df)
print(f"📥 成功读取原始 Excel，总行数: {total_rows} 行")

# ================= 3. 初始化追踪器 =================
reason_missing_data = 0
reason_liftover_failed = 0
reason_duplicate = 0
final_unique_count = 0

hg38_unique_set = set()
attrition_log = [] # 记录阵亡详情列表

print("⏳ 正在逐行审查变异并追踪损耗原因...")

for i, row in df.iterrows():
    # Excel 里的行号通常从 2 开始（有表头）
    excel_row_num = i + 2 
    
    chrom_raw = row.get('chr')
    start_raw = row.get('hg19start')
    ref_raw = row.get('ref')
    alt_raw = row.get('alt')
    gene_name = row.get('gene', 'Unknown')
    
    # ❌ 死因 1：先天残疾 (缺少坐标或碱基信息)
    if pd.isna(chrom_raw) or pd.isna(start_raw) or pd.isna(ref_raw) or pd.isna(alt_raw):
        reason_missing_data += 1
        attrition_log.append({'Excel_Row': excel_row_num, 'Gene': gene_name, 'Status': 'Failed: Missing Info'})
        continue
        
    chrom = str(chrom_raw).replace('chr', '')
    
    # 尝试把 start 转换为整数，如果有奇怪的字符会报错
    try:
        start = int(start_raw)
    except ValueError:
        reason_missing_data += 1
        attrition_log.append({'Excel_Row': excel_row_num, 'Gene': gene_name, 'Status': 'Failed: Invalid Start Pos'})
        continue

    ref = str(ref_raw).upper().strip()
    alt = str(alt_raw).upper().strip()
    
    # ❌ 死因 2：跃迁失败 (在 hg38 中找不到对应坐标，属于基因组更新抛弃的黑户)
    res = lo.convert_coordinate(f"chr{chrom}", start)
    if not res:
        reason_liftover_failed += 1
        attrition_log.append({'Excel_Row': excel_row_num, 'Gene': gene_name, 'Status': 'Failed: LiftOver Unmapped', 'hg19_Pos': start})
        continue
        
    # 构建 hg38 终极物理身份证
    hg38_pos = res[0][1]
    cid = f"{chrom}_{hg38_pos}_{ref}_{alt}"
    
    # ❌ 死因 3：多重影分身 (同一个变异被不同文献/亚型重复收录)
    if cid in hg38_unique_set:
        reason_duplicate += 1
        attrition_log.append({'Excel_Row': excel_row_num, 'Gene': gene_name, 'Status': 'Filtered: Duplicate Variant', 'hg38_ID': cid})
    else:
        # ✅ 活下来的精英，首次出现
        hg38_unique_set.add(cid)
        final_unique_count += 1
        attrition_log.append({'Excel_Row': excel_row_num, 'Gene': gene_name, 'Status': 'Success: Kept Unique', 'hg38_ID': cid})

# ================= 4. 输出侦探报告 =================
print("\n" + "="*55)
print("📊 【gofcards 数据挤水分终极侦探报告】")
print("="*55)
print(f"📂 原始总行数      : {total_rows} 行")
print("-" * 55)
print(f"❌ 死因1 (信息残缺) : - {reason_missing_data} 行 (Excel里缺少REF/ALT/POS)")
print(f"❌ 死因2 (坐标遗失) : - {reason_liftover_failed} 行 (hg19升hg38时无法映射)")
print(f"❌ 死因3 (文献重复) : - {reason_duplicate} 行 (同一个变异被重复写了多行)")
print("-" * 55)
print(f"💎 最终提取纯金变异:   {final_unique_count} 个 (无重复、绝对精确)")
print("="*55)

# ================= 5. 导出追责日志 =================
df_log = pd.DataFrame(attrition_log)
df_log.to_csv(FAILED_LOG_CSV, index=False)
print(f"\n📁 详细追责日志(记录了每一行的死因)已导出至: \n   {FAILED_LOG_CSV}")
print("   你可以下载这个 csv 慢慢核对！")
