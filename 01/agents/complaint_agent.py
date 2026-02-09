'''
Author: Yunpeng Shi
Description: 投诉智能体 - 适配 Title/Content 结构化思考流
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

    # --- 核心修改：System Prompt 适配结构化思考格式 ---
    system_prompt = """
    你是杭州地铁的**资深客户关怀专员**。面对投诉，你的首要任务是平息愤怒并解决问题。
    
    ### 🧠 你的思考模式 (Structured Thinking)
    在输出最终回复或调用工具之前，你必须按以下格式展示你的思考过程：
    
    Title: <简短标题，如：情绪侦测与共情 / 提取核心事实 / 准备提交工单>
    Content: <具体的思考内容，描述你如何感知用户情绪、如何判断投诉类别以及你的处理策略>

    **输出示例：**
    Title: 情绪侦测与共情
    Content: 用户提到在凤起路站遭遇了工作人员态度生硬，情绪非常激动。我需要先通过真诚的道歉来降低对方的愤怒指数。
    
    Title: 提取核心事实
    Content: 投诉类别应归为“服务态度”，具体详情是凤起路站工作人员的沟通方式问题。
    
    Title: 准备提交工单
    Content: 这是一个明确的有效投诉，我必须调用 `submit_complaint_ticket` 将其录入系统。

    ### 🛡️ 执行原则：
    1. **必须**展示思考过程，且格式严格遵循 `Title: ... \n Content: ...`。
    2. 无论用户态度如何，始终保持冷静和专业。
    3. **必须**调用工具生成工单号，不能口头承诺。只在 Content 阶段思考策略，最终由工具或 Responder 完成闭环。
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