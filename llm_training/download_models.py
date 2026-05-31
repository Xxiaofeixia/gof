import os
from huggingface_hub import snapshot_download

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("正在下载 Qwen3-0.6B...")
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./Qwen3-0.6B",
    local_dir_use_symlinks=False
)

print("正在下载 DNA 模型...")
snapshot_download(
    repo_id="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    local_dir="./nt-v2-100m",
    local_dir_use_symlinks=False
)
print("下载完成！")
