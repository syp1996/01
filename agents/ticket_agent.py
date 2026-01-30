from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from state import agentState
from utils import complete_current_task, llm

# --- 第一步：定义工具 (Tools) ---
# 这些是 Agent 的“手脚”。在真实场景中，这里会连接数据库。

@tool
def query_ticket_price(start_station: str, end_station: str) -> str:
    """查询杭州地铁两个站点之间的票价。输入为起始站和终点站名称。"""
    # 模拟数据库查询
    mock_db = {
        ("杭州东站", "武林广场"): "4元",
        ("萧山机场", "武林广场"): "7元",
        ("龙朔", "西湖"): "5元"
    }
    # 双向查询
    price = mock_db.get((start_station, end_station)) or mock_db.get((end_station, start_station))
    if price:
        return f"{start_station} 到 {end_station} 的票价是 {price}。"
    return "抱歉，未查询到该区间的票价信息，请检查站点名称。"

@tool
def query_train_time(station: str) -> str:
    """查询某个站点的首末班车时间。"""
    return f"{station} 的首班车是 06:05，末班车是 22:30。"

# 将工具列表绑定到 LLM
tools = [query_ticket_price, query_train_time]
llm_with_tools = llm.bind_tools(tools)

# --- 第二步：构建 ReAct 微型图 ---

# 定义微型图的状态，仅用于子 Agent 内部循环
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 节点：调用模型
def call_model(state: SubAgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
worker_workflow = StateGraph(SubAgentState)
worker_workflow.add_node("model", call_model)
worker_workflow.add_node("tools", ToolNode(tools)) # LangGraph 内置的工具执行节点

# 设置连线
worker_workflow.add_edge(START, "model")
# 关键：条件边。如果模型决定调用工具 -> 去 tools 节点；如果模型直接说话 -> 结束
worker_workflow.add_conditional_edges("model", tools_condition) 
worker_workflow.add_edge("tools", "model") # 工具执行完，结果回传给模型继续思考

# 编译成可执行对象
react_executor = worker_workflow.compile()


# --- 第三步：主 Agent 节点函数 ---
async def ticket_agent(state: agentState):
    # 1. 从看板获取【纯净输入】
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "ticket_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
            
    if not isolated_input:
        return {"task_board": board} # 防御性编程

    # 2. 构造 System Prompt
    sys_msg = SystemMessage(content="""
    你是票务专家。
    你有权限查询真实的票价和时刻表数据库。
    请根据用户的提问，使用工具查询准确信息。
    不要猜测，必须依据工具返回的结果回答。
    只回答票务问题。
    """)
    
    # 3. 【核心变化】调用 ReAct 微型图进行“思考-行动”循环
    # 我们把 input 包装成消息，丢给 react_executor
    inputs = {"messages": [sys_msg, HumanMessage(content=isolated_input)]}
    
    # await执行，LangGraph 会自动处理多轮工具调用
    result = await react_executor.ainvoke(inputs)
    
    # 4. 获取最终回复
    # ReAct 循环结束后的最后一条消息，就是 LLM 给用户的最终解释
    final_response_content = result["messages"][-1].content
    
    # 5. 销账与返回
    updated_board = complete_current_task(state, "ticket_agent")
    
    return {
        # 注意：这里我们只返回最终结论，不返回中间的工具调用过程，保持主对话历史干净
        "messages": [AIMessage(content=final_response_content, name="ticket_agent")],
        "task_board": updated_board,
        "task_results": {"ticket_agent": final_response_content}
    }