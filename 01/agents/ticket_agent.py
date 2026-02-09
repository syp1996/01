'''
Author: Yunpeng Shi
Description: 票务智能体 - 适配 Title/Content 结构化思考流
'''
import os
from typing import Annotated, List, TypedDict

import utils
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState


# --- Tools 定义 ---
@tool
def query_ticket_balance(card_id: str) -> str:
    """
    查询指定交通卡或乘车码的实时余额。
    Args:
        card_id: 交通卡号或用户唯一识别码。
    """
    # 模拟数据
    return f"【票务系统】卡号 {card_id} 当前余额为：35.50 元。"

@tool
def get_travel_records(card_id: str, count: int = 3) -> str:
    """
    查询指定交通卡最近的乘车记录。
    Args:
        card_id: 交通卡号。
        count: 需要查询的记录条数。
    """
    return f"【票务系统】卡号 {card_id} 最近 2 条记录：\n1. 2026-02-08 08:30 进入凤起路站 - 09:15 离开龙翔桥站 (扣费 4元)\n2. 2026-02-07 17:45 进入火车东站 - 18:30 离开武林广场站 (扣费 5元)"

tools = [query_ticket_balance, get_travel_records]
llm_with_tools = utils.llm.bind_tools(tools)

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

# --- 主 Agent 逻辑改造 ---
async def ticket_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []

    # --- 核心修改：System Prompt 适配结构化思考格式 ---
    system_prompt = """
    你是杭州地铁的**票务服务专家**。你负责处理所有与交通卡余额、充值记录、乘车记录以及票价查询相关的咨询。
    
    ### 🧠 你的思考模式 (Structured Thinking)
    在调用票务系统工具或输出最终结论之前，你必须按以下格式展示你的思考过程。你可以输出一个或多个思考块：
    
    Title: <简短标题，如：分析票务需求 / 验证卡号信息 / 检索系统数据 / 整理交易详情>
    Content: <具体的思考内容，详细描述你如何识别用户想要查什么、如何处理卡号脱敏或补全，以及你的查询策略>

    **输出示例：**
    Title: 分析票务需求
    Content: 用户想要查询账户余额。根据意图，我需要获取用户的卡号或识别码，并调用余额查询接口。
    
    Title: 验证卡号信息
    Content: 历史对话中已包含卡号 A1234567，我可以利用该信息直接进行系统检索。
    
    Title: 检索系统数据
    Content: 正在调用 `query_ticket_balance` 工具以获取该卡号的实时扣费后余额。

    ### 🛡️ 执行原则：
    1. **格式规范**：必须展示思考过程，严格遵守 `Title: ... \n Content: ...`。
    2. **数据准确**：票务信息必须以工具返回的真实数据为准，不得虚构余额或记录。
    3. **安全隐私**：在 Content 思考阶段可以提及卡号，但在最终提供给 Responder 的事实中，注意保护用户隐私。
    """
    
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            *history_context,
            HumanMessage(content=isolated_input)
        ]
    }
    
    # 执行票务处理流程
    result = await react_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    # 更新任务看板
    updated_task = utils.update_task_result(task, result=final_content)
    
    # 计算增量消息
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]

    return {
        "task_board": [updated_task],
        "messages": generated_messages
    }