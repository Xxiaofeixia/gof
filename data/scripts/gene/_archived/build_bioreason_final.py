import pandas as pd
import pysam
from pyliftover import LiftOver
import os
import sys

CHAIN_FILE = "hg19ToHg38.over.chain.gz"
FASTA_FILE = "/gpfs/hpc/home/public/jclabadmin/fasta/Homo_sapiens_assembly38.fasta"
EXCEL_FILE = "gofcards_data_download.xlsx"
VEP_OUTPUT = "my_vep_output.txt"
OUTPUT_FILE = "bioreason_final_result.csv"

def prepare():
    print("正在执行步骤 1：坐标转换 (hg19 -> hg38) 并生成 VCF...")
    lo = LiftOver(CHAIN_FILE)
    df = pd.read_excel(EXCEL_FILE)
    
    vcf_records = []
    for i, row in df.iterrows():
        chrom_raw = str(row['chr']).replace('chr', '')
        res = lo.convert_coordinate(f"chr{chrom_raw}", int(row['hg19start']))
        if res:
            vcf_records.append({
                'CHROM': chrom_raw, 'POS': int(res[0][1]), 'ID': '.',
                'REF': str(row['ref']).upper(), 'ALT': str(row['alt']).upper(),
                'QUAL': '.', 'FILTER': '.', 'INFO': '.'
            })

    df_vcf = pd.DataFrame(vcf_records).sort_values(by=['CHROM', 'POS'])
    with open("to_annotate.vcf", "w") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, row in df_vcf.iterrows():
            f.write(f"{row['CHROM']}\t{row['POS']}\t{row['ID']}\t{row['REF']}\t{row['ALT']}\t{row['QUAL']}\t{row['FILTER']}\t{row['INFO']}\n")
    print(f"成功生成 VCF 文件，包含 {len(df_vcf)} 条变异。")

def clean_score(val):
    """清理多转录本数组，提取第一个有效数字"""
    if pd.isna(val) or val == '-': return '-'
    parts = str(val).replace('&', ',').split(',')
    for p in parts:
        p = p.strip()
        if p and p != '.' and p != '-':
            return p
    return '-'

def get_val(row, keys):
    """根据真实表头精准提取特征"""
    for k in keys:
        if k in row and str(row[k]).strip() not in ['', '-', '.', 'nan']:
            return str(row[k])
    return '-'

def stitch():
    print("正在执行步骤 2：精准对齐 89 列原生特征...")
    
    vep_records = []
    headers = []
    with open(VEP_OUTPUT, 'r') as f:
        for line in f:
            if line.startswith('##'): continue
            if line.startswith('#Uploaded_variation'):
                headers = line.strip().replace('#', '').split('\t')
                continue
            
            if headers:
                fields = line.strip('\n').split('\t')
                if len(fields) < len(headers):
                    fields.extend(['-'] * (len(headers) - len(fields)))
                vep_records.append(dict(zip(headers, fields)))
                
    df_vep = pd.DataFrame(vep_records)
    print(f"成功读取 {len(df_vep)} 条 VEP 原始注释，开始缝合...")

    lo = LiftOver(CHAIN_FILE)
    df_raw = pd.read_excel(EXCEL_FILE)
    fa = pysam.FastaFile(FASTA_FILE)
    final_data = []
    
    for i, row in df_raw.iterrows():
        chrom_raw = str(row['chr']).replace('chr', '')
        ref = str(row['ref']).upper()
        alt = str(row['alt']).upper()
        
        # 定义最终输出格式
        item = {
            'Gene_Symbol': '-', 'Consequence': '-', 'Amino_acids': '-',
            'REVEL_score': '-', 'AlphaMissense': '-', 'CADD_Phred': '-', 'Pscore': '-',
            'gnomAD_EAS_AF': '-', 'ref_seq_4kb': '-', 'alt_seq_4kb': '-'
        }
        
        res = lo.convert_coordinate(f"chr{chrom_raw}", int(row['hg19start']))
        if res:
            p38 = int(res[0][1])
            loc_str = f"{chrom_raw}:{p38}"
            
            matches = df_vep[df_vep['Location'].astype(str).str.startswith(loc_str)]
            
            if not matches.empty:
                # 寻找包含最多分数的转录本作为最优代表
                best_match = matches.iloc[0]
                best_score_count = -1
                
                for _, m in matches.iterrows():
                    am = get_val(m, ['AlphaMissense_score'])
                    rev = get_val(m, ['REVEL_score'])
                    score_count = (1 if clean_score(am) != '-' else 0) + (1 if clean_score(rev) != '-' else 0)
                    if score_count > best_score_count:
                        best_score_count = score_count
                        best_match = m
                
                v = best_match
                
                # 精确对标你刚刚输出的 89 个表头
                item['Gene_Symbol'] = get_val(v, ['SYMBOL'])
                item['Consequence'] = get_val(v, ['Consequence'])
                item['Amino_acids'] = get_val(v, ['Amino_acids'])
                
                item['CADD_Phred'] = clean_score(get_val(v, ['CADD_phred']))
                item['REVEL_score'] = clean_score(get_val(v, ['REVEL_score']))
                item['AlphaMissense'] = clean_score(get_val(v, ['AlphaMissense_score']))
                item['gnomAD_EAS_AF'] = clean_score(get_val(v, ['gnomADe_EAS_AF', 'EAS_AF']))
                
                # Pscore 设为纯数字：优先使用 REVEL，没有则降级使用 AM 或 CADD
                if item['REVEL_score'] != '-':
                    item['Pscore'] = item['REVEL_score']
                elif item['AlphaMissense'] != '-':
                    item['Pscore'] = item['AlphaMissense']
                else:
                    item['Pscore'] = item['CADD_Phred']
                
            try:
                seq_ref = fa.fetch(f"chr{chrom_raw}", p38 - 2001, p38 + 2000).upper()
            except KeyError:
                try: seq_ref = fa.fetch(chrom_raw, p38 - 2001, p38 + 2000).upper()
                except: seq_ref = ""
                
            if len(seq_ref) == 4001:
                item['ref_seq_4kb'] = seq_ref
                item['alt_seq_4kb'] = seq_ref[:2000] + alt + seq_ref[2001:]

        final_data.append(item)

    pd.DataFrame(final_data).to_csv(OUTPUT_FILE, index=False)
    print(f"🎉 任务圆满完成！最终机器学习特征集 {OUTPUT_FILE} 已生成。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "prepare": prepare()
        elif sys.argv[1] == "stitch": stitch()