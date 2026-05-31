import pandas as pd
import os

# 文件路径
original_csv = "BioReason_Task_Dataset.csv"
backup_csv = "BioReason_Task_Dataset_BACKUP_Full.csv"

print("🚀 启动极简过滤引擎：剔除 Neutral 样本...")

# 1. 安全备份原文件（防止数据丢失）
if not os.path.exists(backup_csv):
    os.rename(original_csv, backup_csv)
    print(f"📦 已将原文件备份为: {backup_csv}")
else:
    print(f"📦 备份文件已存在，直接读取...")

# 2. 读取备份的大表
df = pd.read_csv(backup_csv)
print(f"📊 过滤前总数: {len(df)} 条")

# 3. 核心过滤：只保留 Answer 包含 "Pathogenic" 的行（自动剔除 Benign; Neutral）
df_pure = df[df['answer'].str.contains("Pathogenic", na=False)]

# 4. 覆盖保存为原文件名（这样你的训练脚本一句都不用改）
df_pure.to_csv(original_csv, index=False)

print("="*40)
print(f"🏆 提纯完成！")
print(f"💥 现在的 {original_csv} 里只剩 {len(df_pure)} 条纯 GOF 和 LOF 数据！")
print("="*40)
