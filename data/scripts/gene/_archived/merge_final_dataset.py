import pandas as pd
import numpy as np

print("🚀 启动 BIOREASON 终极大一统数据缝合引擎...")

# 专门针对 VEP TXT 格式的读取函数
def load_vep_data(filepath, label):
    print(f"\n⏳ 正在加载 {label} 数据: {filepath}")
    try:
        # 1. 寻找真正的表头行 (避开前面的几十行 # 注释)
        header_line = ""
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#Uploaded_variation'):
                    header_line = line.lstrip('#').strip()
                    break
        
        if not header_line:
            print(f"❌ 警告：找不到表头，尝试强行读取。")
            df = pd.read_csv(filepath, sep='\t', low_memory=False)
        else:
            columns = header_line.split('\t')
            df = pd.read_csv(filepath, sep='\t', comment='#', names=columns, low_memory=False)

        # 2. 极其关键的去重：VEP 可能会为同一个突变输出多个转录本的信息
        # 对于大模型来说，一个突变保留一行最具代表性的特征即可 (keep='first')
        original_len = len(df)
        df = df.drop_duplicates(subset=['Uploaded_variation'], keep='first')
        print(f"   ✅ 加载成功！去重前: {original_len} 行 -> 唯一突变数: {len(df)} 行")

        # 3. 贴上绝对标签
        df['LABEL'] = label
        return df
    
    except Exception as e:
        print(f"❌ 读取 {filepath} 发生致命错误: {e}")
        return pd.DataFrame() # 报错时返回空表防止程序崩溃

# ---------------------------------------------------------
# 请确保这三个文件名与你目录下真实的 VEP 输出文件名完全一致！
# ---------------------------------------------------------
file_gof = "my_vep_output.txt"       # 你的 GOF 特征文件
file_lof = "lof_vep_output.txt"      # 你的 LOF 特征文件
file_neu = "neutral_vep_output.txt"  # 你的 Neutral 特征文件

# 1. 独立读取并打标签
df_gof = load_vep_data(file_gof, 'GOF')
df_lof = load_vep_data(file_lof, 'LOF')
df_neu = load_vep_data(file_neu, 'Neutral')

print("\n🔗 正在执行纵向无缝缝合...")

# 2. 合并三大阵营
df_final = pd.concat([df_gof, df_lof, df_neu], ignore_index=True)

# 3. 洗牌打乱 (Shuffle)
# 这是机器学习的铁律！防止模型训练时前几个 epoch 全是 GOF 导致梯度崩溃
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 导出终极大表
output_name = "BIOREASON_Final_Dataset_All_Features.csv"
df_final.to_csv(output_name, index=False)

print("="*50)
print(f"🏆 历史性时刻！完美合并完成！")
print(f"💾 终极数据集已保存为: {output_name}")
print(f"📊 你的大模型'天平'重量分布如下：")
print(df_final['LABEL'].value_counts())
print("="*50)
