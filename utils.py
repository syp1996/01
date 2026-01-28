import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from state import agentState  # 导入状态定义以便做类型提示

load_dotenv()

# 1. 统一初始化 LLM，避免到处写
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    max_tokens=4096
)

# 2. 定义 Worker 能力描述 (常量)
WORKERS_INFO = {
    "ticket_agent": "处理票务查询、票价计算、线路查询、首末班车时间。",
    "complaint_agent": "处理用户投诉、服务建议、设施故障反馈、失物招领。",
    "general_chat": "处理打招呼、问候、闲聊或无法归类的通用问题。",
    "manager_agent": "处理质检、排班、监控培训相关工作。",
    "judge_agent": "负责热点分析,苗头事件,时序预测,线索筛查相关工作。",
}

# 3. 通用销账辅助函数
def complete_current_task(state: agentState, result: str = None):
    board = state.get("task_board", [])
    current_id = state.get("current_task_id") # 获取令牌
    
    if not current_id:
        return board

    new_board = []
    for task in board:
        t = task.copy()
        # 精准匹配 ID
        if t['id'] == current_id:
            t['status'] = 'done'
            if result:
                t['result'] = result # 将结果回写到看板
        new_board.append(t)
    return new_board