import json
import os
import re  # <--- 新增：引入正则模块用于清洗数据
import sys
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 核心配置 ---
RAW_DOCS_DIR = "./data/raw_docs"
COLLECTION_NAME = "metro_knowledge" 
LOG_FILE = "./data/indexed_files.json"
LOCAL_MODEL_PATH = "./models/bge-small-zh-v1.5" 

# Milvus 配置 (使用 TCP 协议)
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 29530 

# ==========================================
# 🛠️ 工程师优化点 1: 数据清洗函数
# ==========================================
def clean_text_content(text: str) -> str:
    """
    清洗原始文本，去除干扰 RAG 的噪音。
    """
    # 1. 去除页码 (例如 "- 1 -", "Page 1")
    text = re.sub(r'-\s*\d+\s*-', '', text)
    
    # 2. 去除多余的连续换行 (超过2个换行变成2个，保持段落感但去除大片空白)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 3. 去除不可见字符 (如 \u200b 等零宽字符)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text.strip()

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
    # 0. 环境清理
    # ==========================================
    print(">>> [Phase 0] 清理网络代理配置...")
    for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "grpc_proxy", "GRPC_PROXY"]:
        if key in os.environ:
            del os.environ[key]
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"

    # ==========================================
    # 1. 加载本地模型
    # ==========================================
    print(f">>> [Phase 1] 正在从本地路径加载模型: {LOCAL_MODEL_PATH}")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ 错误：找不到模型文件夹 {LOCAL_MODEL_PATH}")
        return

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        print(">>> ✅ 本地模型加载成功！")
    except Exception as e:
        print(f">>> ❌ 模型加载失败: {e}")
        return

    # ==========================================
    # 2. 处理文件 (清洗 + 加载)
    # ==========================================
    processed_log = load_processed_log()
    current_files = get_all_files(RAW_DOCS_DIR)
    
    # ⚠️ 只要有文件，我们就重新构建（为了应用新的切片策略，不再跳过旧文件）
    # 如果你文件特别多，可以恢复增量逻辑，但现在为了调试精度，建议每次全量重跑
    new_files = current_files 
    updated_log = processed_log.copy()

    if not new_files:
        print(">>> 目录中没有文件。")
        return

    print(f">>> 准备处理 {len(new_files)} 个文件...")
    
    docs = []
    for file_path in new_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                # ⚡ 应用优化 1: 清洗文本
                doc.page_content = clean_text_content(doc.page_content)
                
                # ⚡ 应用优化 2: 注入更清晰的元数据
                doc.metadata["source_filename"] = os.path.basename(file_path)
                
                # (可选) 你可以在这里尝试提取 "章节标题" 并加入 metadata，但这需要复杂的规则
            
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"    x 读取失败: {file_path}, {e}")

    if not docs:
        return

    # ==========================================
    # 🛠️ 工程师优化点 2: 优化的切片策略
    # ==========================================
    print(">>> 正在切分文档 (使用优化后的策略)...")
    text_splitter = RecursiveCharacterTextSplitter(
        # 1. 缩小尺寸：350字符通常包含1-2个完整条款，避免包含过多无关噪音
        chunk_size=350,
        # 2. 适度重叠：保证“条款前提”和“具体内容”不会因为切分而断开
        chunk_overlap=50,
        # 3. 增强分隔符：加入中文语义符号，优先级从左到右
        separators=[
            "\n\n", # 优先按段落切
            "\n",   # 其次按行切
            "。",   # 按句号切
            "；",   # 按分号切 (法律条文常用)
            "！", "？", " ", ""
        ]
    )
    splits = text_splitter.split_documents(docs)
    print(f">>> 切分完成，共 {len(splits)} 个高密度切片。")

    # ==========================================
    # 3. 重建 Milvus 集合
    # ==========================================
    milvus_uri = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
    print(f">>> 正在连接 Milvus: {milvus_uri} 并重建集合...")

    try:
        Milvus.from_documents(
            splits,
            embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": milvus_uri, 
                "token": "",
                "timeout": 30
            },
            # ⚠️ 强制清空旧数据，因为切片策略变了，旧向量必须作废
            drop_old=True 
        )
        
        # 更新日志
        for file_path in new_files:
            file_name = os.path.relpath(file_path, RAW_DOCS_DIR)
            updated_log[file_name] = os.path.getmtime(file_path)
        save_processed_log(updated_log)
        
        print(f">>> 🎉 成功！知识库已按照新策略重建完成: {COLLECTION_NAME}")
        
    except Exception as e:
        print(f"\n>>> [错误] 推送失败: {e}")
        if "connect" in str(e).lower():
            print("\n建议排查步骤:")
            print(f"1. 终端执行: nc -zv {MILVUS_HOST} {MILVUS_PORT}")

if __name__ == "__main__":
    build_index()