'''
Author: Yunpeng Shi
Description: 工具类 - 包含 LLM 初始化、日志配置及通用常量
'''
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv
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
    print(f"   当前尝试加载的 .env 路径: {env_path if env_path else '未找到 .env 文件'}")
    print("   请确保 .env 文件中包含: DEEPSEEK_API_KEY=sk-xxxx")
    sys.exit(1)

if not api_base:
    print(f"⚠️ 警告: 未检测到 DEEPSEEK_BASE_URL，将默认使用 DeepSeek 官方地址")
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

# --- 3. 常量定义 (修复报错的关键点) ---
WORKERS_INFO = {
    "ticket_agent": "负责处理地铁票务查询、票价计算、线路规划等任务。",
    "complaint_agent": "负责处理乘客投诉、意见反馈、安检服务态度等问题。",
    "general_chat": "负责处理通用问答、规章制度查询、RAG检索等任务。",
    # 如果你的 Supervisor 还需要调度 manager 或 judge，可以在这里补充，
    # 但通常 WORKERS_INFO 主要用于描述具体的执行单元
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
def update_task_result(task, result):
    """更新任务状态的辅助函数"""
    task['status'] = 'done'
    task['result'] = result
    logger.info(f"任务完成: ID={task.get('id')} 类型={task.get('task_type')}")
    return task