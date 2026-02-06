'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-02-05 12:08:44
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-02-06 15:08:16
FilePath: /general_agent/01/agents/complaint_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi
Description: 投诉智能体 - 引入 CoT 与情感安抚逻辑
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


# --- Tools 定义 ---
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
    ticket_id = f"CPT-{uuid.uuid4().hex[:8].upper()}"
    return f"投诉已成功归档。工单号: {ticket_id}。处理时效: 24小时内。"

tools = [submit_complaint_ticket]
llm_with_tools = llm.bind_tools(tools)

# --- ReAct 微型图 ---
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def call_model(state: SubAgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

worker_workflow = StateGraph(SubAgentState)
worker_workflow.add_node("model", call_model)
worker_workflow.add_node("tools", ToolNode(tools))
worker_workflow.add_edge(START, "model")
worker_workflow.add_conditional_edges("model", tools_condition)
worker_workflow.add_edge("tools", "model")
react_app = worker_workflow.compile()

# --- 主 Agent 逻辑优化 ---
async def complaint_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []

    # 【核心优化】System Prompt 引入 CoT
    system_prompt = """
    你是杭州地铁的**资深客户关怀专员**。面对投诉，你的首要任务是平息愤怒并解决问题。
    
    ### 🧠 深度思考流程 (CoT):
    1. **【情绪侦测】**：用户当前的愤怒指数是多少？（低/中/高）。思考一句最合适的共情话术（例如：“听到这个情况我非常抱歉...”）。
    2. **【关键信息提取】**：从用户的咆哮或描述中提取核心事实 -> `category` (类别) 和 `detail` (详情)。
    3. **【行动执行】**：调用 `submit_complaint_ticket` 工具进行系统录入。
    4. **【闭环反馈】**：拿到工单号后，思考如何用专业且让人放心的语气告知用户。
    
    ### 🛡️ 执行原则：
    - 无论用户态度如何，始终保持冷静和专业。
    - **必须**调用工具生成工单号，不能口头承诺。
    """

    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            *history_context,
            HumanMessage(content=isolated_input)
        ]
    }
    
    result = await react_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    updated_task = update_task_result(task, result=final_content)
    
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]

    return {
        "task_board": [updated_task],
        "messages": generated_messages
    }