import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings  <-- 注释掉这行
from langchain_huggingface import HuggingFaceEmbeddings  # <-- 新增这行
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

RAW_DOCS_DIR = "./data/raw_docs"
VECTOR_DB_DIR = "./data/vector_store"
INDEX_NAME = "metro_knowledge"

def build_index():
    print(">>> 1. 正在加载文档...")
    loader = DirectoryLoader(RAW_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        print("未发现文档，请在 data/raw_docs 下放入 .txt 资料。")
        return

    print(f">>> 加载完成，共 {len(docs)} 个文档。正在切分...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f">>> 切分完成，共生成 {len(splits)} 个切片。")

    print(">>> 2. 正在生成向量 (Local Embeddings)...")
    # --- 修改开始 ---
    # 使用轻量级、效果好的通用模型。会自动下载到本地缓存。
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # --- 修改结束 ---

    print(">>> 3. 构建并保存 FAISS 索引...")
    vector_store = FAISS.from_documents(splits, embedding=embeddings)
    vector_store.save_local(VECTOR_DB_DIR, index_name=INDEX_NAME)
    print(f">>> 索引构建成功！已保存至 {VECTOR_DB_DIR}")

if __name__ == "__main__":
    build_index()