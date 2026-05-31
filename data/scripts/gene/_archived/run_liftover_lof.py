from pyliftover import LiftOver

print("🚀 启动 LOF 坐标升维引擎 (hg19 -> hg38)...")

# 1. 加载你目录下的 chain 字典文件
try:
    lo = LiftOver('hg19ToHg38.over.chain.gz')
    print("✅ 成功加载时空转换字典 (hg19ToHg38)")
except:
    print("❌ 错误：当前目录下找不到 hg19ToHg38.over.chain.gz 文件！")
    exit()

input_vcf = "lof_all_types_to_annotate.vcf"
output_vcf = "lof_hg38_to_annotate.vcf"

converted_count = 0
failed_count = 0

print("⏳ 正在逐行升维坐标...")

with open(input_vcf, 'r') as fin, open(output_vcf, 'w') as fout:
    for line in fin:
        # 保留 VCF 表头
        if line.startswith('#'):
            fout.write(line)
            continue
            
        parts = line.strip().split('\t')
        chrom_raw = parts[0]
        pos_hg19 = int(parts[1])
        
        # pyliftover 需要 'chr' 前缀才能查字典
        chrom_query = chrom_raw if chrom_raw.startswith('chr') else f"chr{chrom_raw}"
        
        # 核心：查询新坐标
        new_coords = lo.convert_coordinate(chrom_query, pos_hg19)
        
        if new_coords and len(new_coords) > 0:
            # 提取转换后的染色体和位置，并去掉 'chr' 保持 VCF 格式清爽
            new_chrom = new_coords[0][0].replace('chr', '')
            new_pos = new_coords[0][1]
            
            # 替换 VCF 行里的坐标
            parts[0] = new_chrom
            parts[1] = str(new_pos)
            # 顺手把 ID 列更新一下，方便后续缝合数据
            parts[2] = f"{new_chrom}_{new_pos}_{parts[3]}_{parts[4]}"
            
            fout.write('\t'.join(parts) + '\n')
            converted_count += 1
        else:
            failed_count += 1

print("="*40)
print(f"🏆 转换完成！")
print(f"🟢 成功穿越到 hg38: {converted_count} 条")
print(f"🔴 坐标在 hg38 中已消失/失效: {failed_count} 条 (这属于正常现象，直接丢弃)")
print(f"💾 终极 LOF 文件已保存为: {output_vcf}")
