'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-26 08:49:23
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/utils.py
Description: 并行化改造版 - 移除全局状态依赖
'''
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. 统一初始化 LLM
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    max_tokens=4096
)

# 2. 定义 Worker 能力描述
WORKERS_INFO = {
    "ticket_agent": "处理票务查询、票价计算、线路查询、首末班车时间。",
    "complaint_agent": "处理用户投诉、服务建议、设施故障反馈、失物招领。",
    "general_chat": "处理打招呼、问候、闲聊或无法归类的通用问题。",
    "manager_agent": "处理质检、排班、监控培训相关工作。",
    "judge_agent": "负责热点分析,苗头事件,时序预测,线索筛查相关工作。",
}

# 3. 并行销账辅助函数 (核心修改)
def update_task_result(task: Dict[str, Any], result: str) -> Dict[str, Any]:
    """
    接收原始任务字典，返回状态更新后的字典。
    用于 Worker 节点返回给 Reducer 进行合并。
    """
    # 创建副本以避免副作用
    updated_task = task.copy()
    
    updated_task['status'] = 'done'
    updated_task['result'] = result
    
    return updated_task