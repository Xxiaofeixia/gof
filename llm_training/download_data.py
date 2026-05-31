import os
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("开始下载 KEGG 数据集到本地...")
try:
    # 下载数据
    dataset = load_dataset("wanglab/kegg", trust_remote_code=True)
    # 保存到本地文件夹
    dataset.save_to_disk("./kegg_data_local")
    print("✅ 数据集已成功保存到 ./kegg_data_local")
except Exception as e:
    print(f"❌ 下载失败: {e}")
