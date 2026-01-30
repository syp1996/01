import os
from typing import Annotated, List, TypedDict

import utils  # ✅ 关键修改：导入整个模块，以便 Mock 和动态调用
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
# --- LangGraph 原生构建模块 ---
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from state import agentState


# ==========================================
# 1. 定义工具 (Tools) - 核心修改点
# ==========================================
@tool
async def lookup_policy(query: str) -> str:
    """
    用于查询地铁相关的规章制度、乘客守则、禁止携带物品、票务政策等官方文档。
    当用户的问题涉及具体规定或政策时，必须调用此工具。
    """
    # ✅ 修复：每次调用时动态获取 Store，确保 Mock 能生效
    store = utils.get_vector_store()
    
    if not store:
        return "知识库系统维护中，暂时无法查询详细规定。"
    
    try:
        # 检索最相关的 3 个片段
        retriever = store.as_retriever(search_kwargs={"k": 3})
        docs = await retriever.ainvoke(query)
        
        if not docs:
            return "未在知识库中找到相关规定。"
        
        results = [f"【相关条款 {i+1}】：{doc.page_content.replace(chr(10), ' ')}" for i, doc in enumerate(docs)]
        return "\n\n".join(results)
    except Exception as e:
        return f"查询出错: {str(e)}"

# 将工具放入列表
tools = [lookup_policy]

# ==========================================
# 2. 手动构建 ReAct 子图 (Sub-Graph)
# ==========================================

# 子图的状态
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ✅ 修正：使用 utils.llm
llm_with_tools = utils.llm.bind_tools(tools)

def call_model(state: SubAgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

rag_workflow = StateGraph(SubAgentState)
rag_workflow.add_node("agent", call_model)
rag_workflow.add_node("tools", ToolNode(tools))
rag_workflow.add_edge(START, "agent")
rag_workflow.add_conditional_edges("agent", tools_condition)
rag_workflow.add_edge("tools", "agent")

rag_app = rag_workflow.compile()

# ==========================================
# 3. 主函数 (Worker Node)
# ==========================================

async def general_chat(state: agentState):
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "general_chat" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
            
    if not isolated_input:
        return {"task_board": board}

    system_prompt = """
    你是一个亲切、专业的地铁综合服务助手。
    策略：
    1. 闲聊不调工具。
    2. 问规定必须调用 lookup_policy。
    """

    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=isolated_input)
        ]
    }
    
    result = await rag_app.ainvoke(inputs)
    final_content = result["messages"][-1].content

    # ✅ 修正：使用 utils.complete_current_task
    updated_board = utils.complete_current_task(state, "general_chat")

    # 注意：这里假设 utils.complete_current_task 返回的是更新后的 board 列表
    # 如果你的 utils 是原地修改或返回 task 对象，请保留你原有的逻辑
    
    # 稍微适配一下你的 update_task_result 逻辑，确保 task 结果被写入
    target_task = next((t for t in board if t['task_type'] == "general_chat"), None)
    if target_task:
        # 尝试使用 task 更新逻辑 (根据你 01 目录下的习惯)
        try:
             utils.update_task_result(target_task, final_content)
        except AttributeError:
             pass # 如果 utils 没有这个函数则跳过，防止报错

    return {
        "messages": [AIMessage(content=final_content, name="general_chat")],
        "task_board": updated_board if isinstance(updated_board, list) else board,
        "task_results": {"general_chat": final_content}
    }    