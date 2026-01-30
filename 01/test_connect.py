import os

from pymilvus import MilvusClient

# 1. 清理代理 (保留这个好习惯)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
for key in ["http_proxy", "https_proxy", "grpc_proxy"]:
    if key in os.environ: del os.environ[key]

# 2. 定义尝试的 URI 列表
uris_to_test = [
    "tcp://127.0.0.1:29530",   # 首选推荐
    "http://127.0.0.1:29530",  # 之前失败的
    "http://localhost:29530",  # 备选
]

print(">>> 开始 Milvus 连接诊断...")

for uri in uris_to_test:
    print(f"\nTesting URI: {uri}")
    try:
        # 尝试建立连接
        client = MilvusClient(uri=uri, token="")
        
        # 尝试一个真实操作来验证连接 (仅建立对象不算成功)
        col_list = client.list_collections()
        
        print(f"✅ 成功连接! 现有集合: {col_list}")
        print(f"👉 请在主代码中使用这个 URI: {uri}")
        break # 成功一个就退出
    except Exception as e:
        print(f"❌ 失败: {e}")