'''
Author: Yunpeng Shi
Description: 修复 Mock 拦截逻辑 (01目录副本)
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
    """查询地铁相关规章制度、乘车守则等官方文档。"""
    
    # ✅ 动态获取，支持 Mock
    store = utils.get_vector_store()
    
    if not store:
        return "系统错误：知识库未正确初始化。"

    try:
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.4}
        )
        docs = await retriever.ainvoke(query)
        
        if not docs:
            return "未在知识库中找到相关规定。"
        
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
    
    system_prompt = "你是一个亲切、专业的地铁综合服务助手。问规定必须调用 lookup_policy。"
    
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
        "messages": [AIMessage(content=final_content, name="general_chat")],
        "task_board": [updated_task]
    }