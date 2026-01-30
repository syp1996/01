import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# 1. 配置 (与构建时保持一致)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
MILVUS_URI = "tcp://127.0.0.1:29530" # 👈 使用刚才验证成功的 TCP 协议
COLLECTION_NAME = "metro_knowledge"

def test_search():
    # --- 环境清理 (保持好习惯) ---
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
    for key in ["http_proxy", "https_proxy", "grpc_proxy"]:
        if key in os.environ: del os.environ[key]

    print(f">>> 1. 加载本地模型: {LOCAL_MODEL_PATH}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径。错误: {e}")
        return

    print(f">>> 2. 连接 Milvus: {MILVUS_URI}")
    try:
        vector_db = Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": MILVUS_URI, 
                "token": "",
            }
        )
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # --- 搜索测试 ---
    # 假设你的文档是《杭州市地铁乘车规则》，我们可以问一个相关问题
    query = "折叠自行车可以带进地铁吗？" 
    
    print(f"\n>>> 3. 正在搜索问题: [{query}] ...")
    
    try:
        # k=3 表示找最相似的 3 条
        results = vector_db.similarity_search(query, k=3)
        
        if not results:
            print("❌ 未找到任何匹配结果 (集合可能是空的？)")
        else:
            print(f"\n✅ 搜索成功！找到 {len(results)} 条相关内容：\n")
            for i, doc in enumerate(results):
                print(f"--- [结果 {i+1}] (来源: {doc.metadata.get('source_filename', '未知')}) ---")
                # 打印内容，去除多余换行
                content_snippet = doc.page_content.replace('\n', ' ')[:150]
                print(f"内容: {content_snippet}...")
                print("------------------------------------------------")
                
    except Exception as e:
        print(f"❌ 搜索过程中出错: {e}")

if __name__ == "__main__":
    test_search()