'''
Author: Yunpeng Shi
Description: 工具类 - 包含 LLM 初始化、向量库连接及通用常量 (修复 RAG 卡顿版)
'''
import logging
import os
import sys
from functools import lru_cache

from dotenv import find_dotenv, load_dotenv
# 新增：引入 HuggingFace 和 Milvus 依赖
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI

# --- 1. 环境变量加载 ---
env_path = find_dotenv()
load_dotenv(env_path, override=True)

# 读取配置
api_key = os.getenv("DEEPSEEK_API_KEY") 
api_base = os.getenv("DEEPSEEK_BASE_URL")

# 检查配置
if not api_key:
    print(f"❌ 致命错误: 未检测到 DEEPSEEK_API_KEY！")
    sys.exit(1)

if not api_base:
    api_base = "https://api.deepseek.com"

# --- 2. 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler("agent_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MetroAgent")

# --- 3. 常量定义 ---
WORKERS_INFO = {
    "ticket_agent": "负责处理地铁票务查询、票价计算、线路规划等任务。",
    "complaint_agent": "负责处理乘客投诉、意见反馈、安检服务态度等问题。",
    "general_chat": "负责处理通用问答、规章制度查询、RAG检索等任务。",
}

# --- 4. LLM 初始化 ---
try:
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0,
        max_retries=3,
        timeout=60,
    )
    logger.info(f"LLM 初始化成功 (Base: {api_base})")
except Exception as e:
    logger.error(f"LLM 初始化失败: {e}")
    raise e

# --- 5. 辅助函数 ---

# 【核心修复 1】使用全局变量或 lru_cache 缓存 Embedding 模型
# 防止每次请求都重新加载模型导致卡顿
_cached_embeddings = None

def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings:
        return _cached_embeddings
    
    # 优先尝试本地模型路径
    # 注意：确保 download_bge.py 下载的路径与此一致
    model_path = "./models/bge-small-zh-v1.5"
    
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ 本地模型未找到: {model_path}，尝试使用默认 all-MiniLM-L6-v2 (可能需要下载)")
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
    
    logger.info(f"正在加载 Embedding 模型: {model_path} ...")
    try:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}, # Docker 内通常用 CPU
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Embedding 模型加载完成")
        return _cached_embeddings
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return None

def update_task_result(task, result):
    """更新任务状态的辅助函数"""
    task['status'] = 'done'
    task['result'] = result
    logger.info(f"任务完成: ID={task.get('id')} 类型={task.get('task_type')}")
    return task

def get_vector_store():
    """
    获取 Milvus 向量数据库实例 (Docker 适配版)
    """
    embeddings = get_embeddings()
    if not embeddings:
        return None

    # 【核心修复 2】正确获取 Docker 内部的网络地址
    # 在 docker-compose.yml 中，milvus 主机名就是 'milvus'
    # 内部端口是 19530 (不要用外部的 29530)
    milvus_host = os.getenv("MILVUS_HOST", "milvus") 
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    
    # 构建连接 URI
    connection_args = {
        "uri": f"tcp://{milvus_host}:{milvus_port}",
        "token": "", # 如果没有设密码留空
        "timeout": 5 # 设置超时时间，防止无限卡死
    }
    
    collection_name = "metro_knowledge"
    
    try:
        # 尝试连接
        vector_db = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            auto_id=True
        )
        return vector_db
    except Exception as e:
        logger.error(f"❌ 向量库连接失败 (Host: {milvus_host}:{milvus_port}): {e}")
        return None