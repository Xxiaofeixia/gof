import os
from huggingface_hub import snapshot_download

print("🚀 启动 Hugging Face 高速下载引擎...")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
BASE_DIR = "/gpfs/hpc/home/lijc/mapengtao/Bioreason/BioReason/pretrained_models"
os.makedirs(BASE_DIR, exist_ok=True)

# 【核心修正】：名字一字不差地对齐 Hugging Face 官方仓库
models_to_download = {
    # 你的文本大模型
    "Qwen3-4B": "Qwen/Qwen2.5-3B", 
    # 你的 DNA 模型一
    "NT-v2-500m": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    # 你的 DNA 模型二 (修正版，加上了 _base)
    "evo2-1b": "arcinstitute/evo2_1b_base" 
}

for local_folder_name, repo_id in models_to_download.items():
    save_path = os.path.join(BASE_DIR, local_folder_name)
    print(f"\n==================================================")
    print(f"⏳ 正在拉取: {repo_id}")
    print(f"💾 保存路径: {save_path}")
    print(f"==================================================")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
            resume_download=True, # 依然开启断点续传，之前下了一半的 Qwen 会瞬间接上！
            max_workers=8
        )
        print(f"🏆 {local_folder_name} 下载成功！")
    except Exception as e:
        print(f"❌ 下载 {local_folder_name} 时发生错误: {e}")

print("\n🎉 所有模型下载任务结束！")
