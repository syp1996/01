'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:57:25
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/ticket_agent.py
Description: 并行化改造版 - 修复旧引用报错
'''
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState  # 【关键】使用 WorkerState
from utils import (  # 【关键】使用 update_task_result (不再引用 complete_current_task)
    llm, update_task_result)


# --- Tools 定义 (保持不变) ---
@tool
def query_ticket_price(start_station: str, end_station: str) -> str:
    """查询杭州地铁两个站点之间的票价。输入为起始站和终点站名称。"""
    mock_db = {
        ("杭州东站", "武林广场"): "4元",
        ("萧山机场", "武林广场"): "7元",
        ("龙朔", "西湖"): "5元"
    }
    price = mock_db.get((start_station, end_station)) or mock_db.get((end_station, start_station))
    if price:
        return f"{start_station} 到 {end_station} 的票价是 {price}。"
    return "抱歉，未查询到该区间的票价信息，请检查站点名称。"

@tool
def query_train_time(station: str) -> str:
    """查询某个站点的首末班车时间。"""
    return f"{station} 的首班车是 06:05，末班车是 22:30。"

tools = [query_ticket_price, query_train_time]
llm_with_tools = llm.bind_tools(tools)

# --- ReAct 微型图定义 (保持不变) ---
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def call_model(state: SubAgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

worker_workflow = StateGraph(SubAgentState)
worker_workflow.add_node("model", call_model)
worker_workflow.add_node("tools", ToolNode(tools))
worker_workflow.add_edge(START, "model")
worker_workflow.add_conditional_edges("model", tools_condition) 
worker_workflow.add_edge("tools", "model")
react_executor = worker_workflow.compile()


# --- 主 Agent 节点函数 (核心修复) ---
async def ticket_agent(state: WorkerState):
    """
    接收 WorkerState (包含 task 字典)，返回更新后的 task_board 列表。
    """
    
    # 1. 直接获取任务 (无需遍历)
    task = state["task"]
    isolated_input = task['input_content']

    # 【获取历史】
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []
    
    print(f"[Ticket] 正在处理: {isolated_input}")

    # 2. 构造 System Prompt
    sys_msg = SystemMessage(content="""
    你是票务专家。
    你有权限查询真实的票价和时刻表数据库。
    请根据用户的提问，使用工具查询准确信息。
    不要猜测，必须依据工具返回的结果回答。
    只回答票务问题。
    """)
    
    # 3. 执行微型图
    # 【注入历史】
    inputs = {
        "messages": [sys_msg] + history_context + [HumanMessage(content=isolated_input)]
    }
    result = await react_executor.ainvoke(inputs)
    final_response_content = result["messages"][-1].content
    
    # 4. 销账 (使用新函数 update_task_result)
    updated_task = update_task_result(task, result=final_response_content)
    
    # 5. 返回结果 (通过 Reducer 合并)
    return {
        "messages": [AIMessage(content=final_response_content, name="ticket_agent")],
        "task_board": [updated_task] 
    }