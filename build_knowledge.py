import json
import os
import time
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 配置路径 ---
RAW_DOCS_DIR = "./data/raw_docs"
VECTOR_DB_DIR = "./data/vector_store"
INDEX_NAME = "metro_knowledge"
LOG_FILE = "./data/indexed_files.json"  # 新增：用于记录已索引文件的日志

def load_processed_log() -> Dict[str, float]:
    """加载已处理文件的记录 (文件名: 最后修改时间)"""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_processed_log(log_data: Dict[str, float]):
    """保存已处理文件的记录"""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def get_all_files(directory: str, ext: str = ".txt") -> List[str]:
    """获取目录下所有指定后缀的文件路径"""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                file_paths.append(os.path.join(root, file))
    return file_paths

def build_index():
    print(">>> [增量构建] 正在检查文件变更...")
    
    # 1. 初始化 Embedding 模型 (必须与查询时一致)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. 加载或初始化向量库
    vector_store = None
    if os.path.exists(VECTOR_DB_DIR) and os.path.exists(os.path.join(VECTOR_DB_DIR, f"{INDEX_NAME}.faiss")):
        try:
            print(f">>> 发现现有索引，正在加载: {VECTOR_DB_DIR}")
            vector_store = FAISS.load_local(
                VECTOR_DB_DIR, 
                embeddings, 
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f">>> 现有索引加载失败 ({e})，将重新构建。")
    else:
        print(">>> 未发现现有索引，将新建向量库。")

    # 3. 对比文件状态，找出需要新增的文件
    processed_log = load_processed_log()
    current_files = get_all_files(RAW_DOCS_DIR)
    
    new_files = []
    updated_log = processed_log.copy()
    
    for file_path in current_files:
        # 获取文件最后修改时间
        mtime = os.path.getmtime(file_path)
        file_name = os.path.relpath(file_path, RAW_DOCS_DIR) # 存相对路径
        
        # 判断条件：文件不在日志中，或者文件的修改时间比日志里的新
        # 注意：FAISS 本地版不支持简单的“更新/删除”操作。
        # 为了简单起见，这里我们主要处理【新增文件】。
        # 如果文件被修改了，简单的追加会导致重复。
        # 工业级方案通常需要 ID 管理，这里我们采用“只处理未记录的新文件”策略。
        if file_name not in processed_log:
            new_files.append(file_path)
            updated_log[file_name] = mtime
        # 如果想处理修改过的文件，需要先从库里删除旧 ID（复杂），或者建议用户定期全量重构。

    if not new_files:
        print(">>> 没有发现新文件，无需更新。")
        return

    print(f">>> 发现 {len(new_files)} 个新文件，准备处理...")
    for f in new_files:
        print(f"    + {os.path.basename(f)}")

    # 4. 加载并切分新文件
    docs = []
    for file_path in new_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"    x 加载文件失败 {file_path}: {e}")

    if not docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f">>> 切分完成，共生成 {len(splits)} 个新切片。正在向量化...")

    # 5. 更新向量库 (Add vs Create)
    if vector_store:
        # 如果库已存在，使用 add_documents 追加
        vector_store.add_documents(splits)
        print(">>> 新切片已追加到现有索引。")
    else:
        # 如果库不存在，使用 from_documents 创建
        vector_store = FAISS.from_documents(splits, embedding=embeddings)
        print(">>> 已创建新索引。")

    # 6. 保存索引和日志
    vector_store.save_local(VECTOR_DB_DIR, index_name=INDEX_NAME)
    save_processed_log(updated_log)
    
    print(f">>> 更新完成！索引已保存至 {VECTOR_DB_DIR}")

if __name__ == "__main__":
    build_index()