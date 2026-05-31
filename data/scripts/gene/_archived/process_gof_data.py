import pandas as pd
import pysam
from pyliftover import LiftOver
import os

# --- 路径配置 ---
CHAIN_FILE = "hg19ToHg38.over.chain.gz"
FASTA_FILE = "/gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta"
EXCEL_FILE = "gofcards_data_download.xlsx"
SNV_FILE = "SNV.txt"
OUTPUT_FILE = "bioreason_gold_dataset_4kb.csv"

def main():
    print("1. 正在初始化资源 (LiftOver & Fasta)...")
    if not os.path.exists(CHAIN_FILE):
        print(f"错误: 找不到链文件 {CHAIN_FILE}，请先上传。")
        return
    lo = LiftOver(CHAIN_FILE)
    fa = pysam.FastaFile(FASTA_FILE)
    
    # 2. 读取并预处理数据
    print("2. 正在读取并清洗原始数据...")
    df_excel = pd.read_excel(EXCEL_FILE)
    # 统一染色体格式
    df_excel['chr_std'] = df_excel['chr'].astype(str).str.replace('chr', '', case=False)
    
    # 3. 读取并去重 SNV 注释库
    print("3. 正在预处理 SNV.txt (去除重复转录本)...")
    # low_memory=False 防止大文件读取时的类型警告
    df_snv = pd.read_csv(SNV_FILE, sep='\t', low_memory=False)
    df_snv['chr_std'] = df_snv['Chr'].astype(str).str.replace('chr', '', case=False)
    # 每个基因组位置只保留一行核心特征
    df_snv_unique = df_snv.drop_duplicates(subset=['chr_std', 'Start', 'Ref', 'Alt'], keep='first')
    
    print(f"注释库去重完成: 原始 {len(df_snv)} 行 -> 唯一位点 {len(df_snv_unique)} 行")

    # 4. 核心匹配逻辑
    print("4. 正在执行坐标转换与特征缝合 (hg19 -> hg38)...")
    final_results = []
    success_count = 0
    
    for i, row in df_excel.iterrows():
        # 安全获取碱基并转大写
        ref_in = str(row['ref']).upper() if pd.notna(row['ref']) else ""
        alt_in = str(row['alt']).upper() if pd.notna(row['alt']) else ""
        if not ref_in or ref_in == "NAN": continue

        # Liftover 转换
        c_long = f"chr{row['chr_std']}"
        res = lo.convert_coordinate(c_long, int(row['hg19start']))
        if not res: continue
        hg38_p = res[0][1]
        
        # 在 SNV 库中寻找匹配 (尝试原位和偏移位)
        match = df_snv_unique[
            (df_snv_unique['chr_std'] == row['chr_std']) & 
            (df_snv_unique['Start'] == hg38_p) & 
            (df_snv_unique['Ref'].astype(str).str.upper() == ref_in)
        ]
        
        if match.empty: # 尝试 -1 偏移匹配 0-based 坐标
            match = df_snv_unique[
                (df_snv_unique['chr_std'] == row['chr_std']) & 
                (df_snv_unique['Start'] == hg38_p - 1) & 
                (df_snv_unique['Ref'].astype(str).str.upper() == ref_in)
            ]

        if not match.empty:
            # 5. 提取 4001bp 序列
            try:
                feat = match.iloc[0].to_dict()
                actual_p = int(feat['Start'])
                
                # 提取 (变异点在中心)
                start, end = actual_p - 2001, actual_p + 2000
                ref_seq = fa.fetch(c_long, start, end).upper()
                
                if len(ref_seq) == 4001 and ref_seq[2000] == ref_in:
                    # 缝合所有模态：Excel 背景 + SNV 预测分 + DNA 序列
                    combined = {**row.to_dict(), **feat}
                    combined['hg38_pos_final'] = actual_p
                    combined['ref_seq_4kb'] = ref_seq
                    combined['alt_seq_4kb'] = ref_seq[:2000] + alt_in + ref_seq[2001:]
                    final_results.append(combined)
                    success_count += 1
            except Exception:
                continue

    # 6. 保存最终结果
    if final_results:
        out_df = pd.DataFrame(final_results)
        # 移除多余的重复列
        if 'chr_std' in out_df.columns: out_df = out_df.drop(columns=['chr_std'])
        out_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n🎉 任务圆满完成！")
        print(f"生成文件: {OUTPUT_FILE}")
        print(f"最终有效样本数: {len(out_df)} / 原始总数: {len(df_excel)}")
    else:
        print("❌ 匹配失败：未找到符合条件的对齐数据，请检查坐标版本。")

if __name__ == "__main__":
    main()