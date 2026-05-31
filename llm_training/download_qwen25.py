import os
from huggingface_hub import snapshot_download

# 显式指定镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("正在通过镜像站下载原生兼容的 Qwen2.5-0.5B...")
try:
    snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B",
        local_dir="./Qwen2.5-0.5B",
        resume_download=True,
        max_workers=4,
        endpoint="https://hf-mirror.com"  # 强制指定终结点
    )
    print("✅ 下载完成！")
except Exception as e:
    print(f"❌ 下载失败，请检查网络。错误详情: {e}")
