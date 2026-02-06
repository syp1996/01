'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/complaint_agent.py
Description: 并行化 ReAct 改造版 - 增加工单录入能力 + 增加思考过程持久化
'''
import uuid
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState
from utils import llm, update_task_result


# --- 1. 定义工具 (Tools) ---
@tool
def submit_complaint_ticket(category: str, detail: str) -> str:
    """
    将用户的投诉内容录入后台系统，并生成唯一的工单号。
    Args:
        category: 投诉类别（如：服务态度、设备故障、环境卫生）。
        detail: 投诉的具体详情描述。
    Returns:
        包含工单号的确认信息。
    """
    # 模拟数据库操作
    ticket_id = f"CPT-{uuid.uuid4().hex[:8].upper()}"
    print(f"\n[System] 📝 投诉已录入数据库: ID={ticket_id} | 类型={category}")
    return f"投诉已成功归档。工单号: {ticket_id}。处理时效: 24小时内。"

tools = [submit_complaint_ticket]

# --- 2. 构建 ReAct 子图 ---
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools(tools)

def call_model(state: SubAgentState):
    # 调用模型，模型会自动决定是否使用工具
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 定义图结构：Model -> Tools -> Model (循环)
worker_workflow = StateGraph(SubAgentState)
worker_workflow.add_node("model", call_model)
worker_workflow.add_node("tools", ToolNode(tools))

worker_workflow.add_edge(START, "model")
worker_workflow.add_conditional_edges("model", tools_condition)
worker_workflow.add_edge("tools", "model")

react_app = worker_workflow.compile()

# --- 3. 主 Agent 函数 ---
async def complaint_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    
    print(f"[Complaint] 正在处理 (ReAct): {isolated_input}")

    # 获取全局历史，保持上下文连贯
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []

    # System Prompt：强制要求使用工具
    system_prompt = """
    你是杭州地铁的资深客户投诉专员。
    你的职责不仅仅是安抚用户，更重要的是**切实解决问题**。
    
    ### 核心流程：
    1. **安抚情绪**：首先对用户的不愉快经历表示歉意。
    2. **执行录入**：必须调用 `submit_complaint_ticket` 工具，将投诉详情录入系统。
    3. **反馈结果**：将工具生成的【工单号】反馈给用户，让用户感到放心。
    
    请确保语气诚恳、专业。
    """

    # 构造输入：System + History + Current Input
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            *history_context,
            HumanMessage(content=isolated_input)
        ]
    }
    
    # 执行 ReAct 循环
    result = await react_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    # 更新任务状态
    updated_task = update_task_result(task, result=final_content)
    
    # =========== 【新增】 计算需要持久化的思考过程消息 ===========
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]
    # ========================================================

    return {
        "task_board": [updated_task],
        # 【关键修复】
        "messages": generated_messages
    }