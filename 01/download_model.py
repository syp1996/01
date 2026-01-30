import os

from huggingface_hub import snapshot_download

# 定义下载目录
local_model_dir = "./models/all-MiniLM-L6-v2"

print(f"正在下载模型到: {local_model_dir} ...")

# 下载模型所有文件到指定目录
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=local_model_dir,
    local_dir_use_symlinks=False  # 确保下载的是真实文件，不是快捷方式
)

print("✅ 下载完成！现在你可以关掉 VPN 了。")