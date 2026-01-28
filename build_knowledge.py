import json
import os
import sys
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 配置 ---
RAW_DOCS_DIR = "./data/raw_docs"
COLLECTION_NAME = "metro_knowledge" 
LOG_FILE = "./data/indexed_files.json"
# 指向刚才下载的本地文件夹路径
LOCAL_MODEL_PATH = "./models/bge-small-zh-v1.5" 

# Milvus 配置
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 29530 

def load_processed_log() -> Dict[str, float]:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_processed_log(log_data: Dict[str, float]):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def get_all_files(directory: str, ext: str = ".txt") -> List[str]:
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                file_paths.append(os.path.join(root, file))
    return file_paths

def build_index():
    # ==========================================
    # 0. 环境清理 (最先执行，防止干扰)
    # ==========================================
    print(">>> [Phase 0] 清理网络代理配置...")
    # 强力清理所有代理变量
    for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "grpc_proxy", "GRPC_PROXY"]:
        if key in os.environ:
            del os.environ[key]
    
    # 设置不走代理的名单
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"
    print("    - 代理环境变量已清理，确保直连 Docker。")

    # ==========================================
    # 1. 加载本地模型 (不联网)
    # ==========================================
    print(f">>> [Phase 1] 正在从本地路径加载模型: {LOCAL_MODEL_PATH}")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ 错误：找不到模型文件夹 {LOCAL_MODEL_PATH}")
        print("   请先运行 download_model.py 下载模型！")
        return

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH, # 👈 直接传文件夹路径
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        print(">>> ✅ 本地模型加载成功！")
    except Exception as e:
        print(f">>> ❌ 模型加载失败: {e}")
        return

    # ==========================================
    # 2. 处理文件
    # ==========================================
    processed_log = load_processed_log()
    current_files = get_all_files(RAW_DOCS_DIR)
    new_files = []
    updated_log = processed_log.copy()

    for file_path in current_files:
        mtime = os.path.getmtime(file_path)
        file_name = os.path.relpath(file_path, RAW_DOCS_DIR)
        if file_name not in processed_log: # 简单逻辑：只看文件名是否记录过
            new_files.append(file_path)
            updated_log[file_name] = mtime

    if not new_files:
        print(">>> 没有发现新文件，无需更新。")
        return

    print(f">>> 发现 {len(new_files)} 个新文件，准备处理...")
    
    docs = []
    for file_path in new_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            # 优化：增加 source 元数据
            for doc in loaded_docs:
                doc.metadata["source_filename"] = os.path.basename(file_path)
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"    x 读取失败: {file_path}, {e}")

    if not docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f">>> 切分完成，共 {len(splits)} 个切片。")

    # ==========================================
    # 3. 推送到 Milvus
    # ==========================================
    
    # ⚠️ 关键修改：URI 格式必须带 http://
    milvus_uri = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}" 
    
    print(f">>> 正在连接 Milvus: {milvus_uri}")

    try:
        Milvus.from_documents(
            splits,
            embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": milvus_uri,  # 结果: tcp://127.0.0.1:29530
                "token": "",
                "timeout": 30
            },
            drop_old=True 
        )
        
        save_processed_log(updated_log)
        print(f">>> 成功！数据已写入 Milvus 集合: {COLLECTION_NAME}")
        
    except Exception as e:
        print(f"\n>>> [错误] 推送失败: {e}")
        # 如果是连接错误，打印更详细的提示
        if "connect" in str(e).lower():
            print("\n建议排查步骤:")
            print(f"1. 终端执行: nc -zv {MILVUS_HOST} {MILVUS_PORT}")
            print("2. 确保 VPN 已彻底关闭")
            print("3. 确保 Docker 容器正在运行 (docker ps)")

if __name__ == "__main__":
    build_index()