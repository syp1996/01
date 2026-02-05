'''
Author: Yunpeng Shi
Description: 修复 Mock 拦截逻辑 (01目录副本) - 修复死循环版
'''
import os
from typing import Annotated, List, TypedDict

import utils  # ✅ 导入整个 utils
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState


@tool
async def lookup_policy(query: str) -> str:
    """
    检索地铁乘客守则、禁止携带物品、票务规定等【书面规章制度】。
    注意：不要用此工具查询线路数量、站点信息等事实性知识。
    """
    
    # ✅ 动态获取，支持 Mock
    store = utils.get_vector_store()
    
    if not store:
        return "系统提示：知识库服务暂时不可用，请直接根据常识回答。"

    try:
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.4}
        )
        docs = await retriever.ainvoke(query)
        
        if not docs:
            return "【检索结果】知识库中未包含相关具体规定。请你基于通用知识回答用户，不要再次尝试检索。"
        
        results = []
        for i, doc in enumerate(docs):
            # [修改] 修复 Python 3.11 f-string 不支持反斜杠的问题
            # 将 replace 操作移出 f-string
            clean_content = doc.page_content.replace('\n', ' ')
            results.append(f"【条款 {i+1}】: {clean_content}")
            
        return "\n\n".join(results)
    except Exception as e:
        return f"系统错误：知识库检索失败 ({str(e)})。"

tools = [lookup_policy]
llm_with_tools = utils.llm.bind_tools(tools)

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

async def general_chat(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    global_messages = state.get("messages", [])
    
    # --- 核心修改：防死循环 Prompt ---
    system_prompt = """
    你是杭州地铁的综合服务助手。
    
    ### 决策逻辑：
    1. **判断问题类型**：
       - 如果用户问的是**“能不能带”、“罚款多少”、“票务规则”**等政策类问题 -> **必须调用 `lookup_policy`**。
       - 如果用户问的是**“有多少条线”、“某站的首班车”**等事实/常识类问题 -> **禁止调用工具**，直接用你的模型知识回答。
       
    2. **防死循环机制**：
       - 如果调用了一次 `lookup_policy` 且返回“未找到”，**严禁再次调用该工具**。
       - 此时应直接回复：“抱歉，规章库中暂时没有相关记录，但根据一般经验……”，或者直接基于你的常识给出建议。
    """
    
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            *global_messages[:-1],
            HumanMessage(content=isolated_input)
        ]
    }
    
    result = await react_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    # 使用 utils 更新
    updated_task = utils.update_task_result(task, result=final_content)
    
    return {
        # 只返回任务看板更新，防止污染历史
        "task_board": [updated_task]
    }