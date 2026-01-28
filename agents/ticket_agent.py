from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from state import agentState
from utils import complete_current_task, llm


# --- Tools 定义保持不变 ---
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

# --- ReAct 微型图定义保持不变 ---
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


# --- 主 Agent 节点函数 (核心修改) ---
async def ticket_agent(state: agentState):
    # 1. 获取令牌和看板
    current_id = state.get("current_task_id")
    board = state.get("task_board", [])
    
    # 2. 凭令牌取数据
    target_task = None
    for task in board:
        if task['id'] == current_id:
            target_task = task
            break
            
    if not target_task:
        # 防御性编程：如果没有找到对应ID的任务，直接返回
        return {"task_board": board}

    isolated_input = target_task['input_content']

    # 3. 构造 System Prompt
    sys_msg = SystemMessage(content="""
    你是票务专家。
    你有权限查询真实的票价和时刻表数据库。
    请根据用户的提问，使用工具查询准确信息。
    不要猜测，必须依据工具返回的结果回答。
    只回答票务问题。
    """)
    
    # 4. 执行微型图
    inputs = {"messages": [sys_msg, HumanMessage(content=isolated_input)]}
    result = await react_executor.ainvoke(inputs)
    final_response_content = result["messages"][-1].content
    
    # 5. 销账与返回 (传入结果)
    updated_board = complete_current_task(state, result=final_response_content)
    
    return {
        "messages": [AIMessage(content=final_response_content, name="ticket_agent")],
        "task_board": updated_board
        # 注意：不再返回 task_results
    }