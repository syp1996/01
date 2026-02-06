'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-02-05 12:08:44
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-02-06 13:51:02
FilePath: /general_agent/01/agents/ticket_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi
Description: 票务智能体 - 引入 CoT 思维链与深度思考
'''
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState
from utils import llm, update_task_result


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

# --- ReAct 微型图定义 ---
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
react_executor = worker_workflow.compile()

# --- 主 Agent 逻辑优化 ---
async def ticket_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []
    
    # 【核心优化】System Prompt 引入 CoT
    sys_msg = SystemMessage(content="""
    你是杭州地铁的**票务与行程专家**。你的职责是提供精准的出行信息。

    ### 🧠 深度思考流程 (CoT):
    1. **【站点核对】**：首先分析用户输入的站点名称是否清晰？(例如 "东站" 指的是 "杭州东站")。
    2. **【意图确认】**：用户是问票价、时间还是路线？
    3. **【工具决策】**：
       - 问票价 -> 调用 `query_ticket_price`
       - 问首末班 -> 调用 `query_train_time`
    4. **【结果验证】**：工具返回结果后，检查是否合理。如果未查到，思考是否需要提示用户检查站名。
    5. **关键格式要求：**
    思考完成后，必须输出 `=====FINAL_ANSWER=====`，然后紧接着输出票价或时间的具体数字/信息。

    ### 🛡️ 约束：
    - 严禁猜测票价或时间，必须以工具返回结果为准。
    - 回复要简洁明了，直接给出数字。
    """)
    
    inputs = {
        "messages": [sys_msg] + history_context + [HumanMessage(content=isolated_input)]
    }
    
    result = await react_executor.ainvoke(inputs)
    final_response_content = result["messages"][-1].content
    
    updated_task = update_task_result(task, result=final_response_content)
    
    # 计算增量消息 (用于前端展示思考过程)
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]

    return {
        "task_board": [updated_task],
        "messages": generated_messages
    }