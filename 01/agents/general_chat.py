'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/general_chat.py
Description: 并行化改造版 - 修复 Mock 拦截逻辑
'''
import os
from typing import Annotated, List, TypedDict

# 1. 导入 utils 模块
import utils
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# 从 utils 导入必要的函数和对象
from utils import complete_current_task, llm, update_task_result

from state import WorkerState


@tool
async def lookup_policy(query: str) -> str:
    """查询地铁相关规章制度、乘车守则等官方文档。"""
    
    # ✅ 核心修正：显式调用 utils 模块里的函数
    # 这样测试代码里的 @patch("utils.get_vector_store") 才能 100% 拦截成功
    store = utils.get_vector_store()
    
    if not store:
        return "系统错误：知识库未正确初始化。"

    try:
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5, 
                "score_threshold": 0.4,
                "param": {"metric_type": "L2", "nprobe": 10} 
            }
        )
        
        docs = await retriever.ainvoke(query)
        
        if not docs:
            return "未在知识库中找到相关规定。"
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source_filename', '未知')
            clean_content = doc.page_content.replace('\n', ' ')
            results.append(f"【条款 {i+1}】(来源: {source}): {clean_content}")
            
        return "\n\n".join(results)
    except Exception as e:
        return f"系统错误：知识库检索失败 ({str(e)})。"

# --- 2. ReAct 子图定义 ---

tools = [lookup_policy]
llm_with_tools = llm.bind_tools(tools)

class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def call_model(state: SubAgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

rag_workflow = StateGraph(SubAgentState)
rag_workflow.add_node("agent", call_model)
rag_workflow.add_node("tools", ToolNode(tools))
rag_workflow.add_edge(START, "agent")
rag_workflow.add_conditional_edges("agent", tools_condition)
rag_workflow.add_edge("tools", "agent")
rag_app = rag_workflow.compile()

# --- 3. 主函数 ---

async def general_chat(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []

    system_prompt = """你是一个亲切、专业的地铁综合服务助手。
    你的主要职责是陪乘客闲聊，或者依据真实规定解答地铁政策问题。

    ### 核心指令：
    1. **必须查证**：涉及地铁政策问题必须调用 lookup_policy。
    2. **强制标记**：引用知识库信息必须在句末加 `【📚知识库】`。
    """

    inputs = {
        "messages": [SystemMessage(content=system_prompt)] + history_context + [HumanMessage(content=isolated_input)]
    }
    result = await rag_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    return {
        "messages": [AIMessage(content=final_content, name="general_chat")],
        "task_board": [update_task_result(task, result=final_content)]
    }    