import os

from huggingface_hub import snapshot_download

# 1. 定义新模型的 ID 和本地存放位置
repo_id = "BAAI/bge-small-zh-v1.5"
local_model_dir = "./models/bge-small-zh-v1.5"

print(f"🚀 正在下载中文强力模型: {repo_id} ...")
print("   (这可能需要几分钟，请保持 VPN 开启)")

# 2. 执行下载
snapshot_download(
    repo_id=repo_id,
    local_dir=local_model_dir,
    local_dir_use_symlinks=False, # 确保下载的是真实文件
    # BGE 有些文件很大，我们可以排除不需要的训练文件，只下推理用的
    ignore_patterns=["*.msgpack", "model.safetensors", "*.h5", "*.ot"] 
    # 注意：pytorch_model.bin 是必须的，safetensors 有时候 LangChain 支持不好，保守起见下 bin
)

print(f"✅ 下载完成！模型已保存在: {local_model_dir}")
print("👉 下一步：请更新你的代码，将 model_name 指向这个新路径。")