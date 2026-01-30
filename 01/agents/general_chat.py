'''
Author: Yunpeng Shi
Description: 修复 Mock 拦截逻辑
'''
import os
from typing import Annotated, List, TypedDict

import utils  # ✅ 关键：只导入整个模块
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState

# ⚠️ 注意：不要在这里写 from utils import get_vector_store，否则 Mock 会失效

@tool
async def lookup_policy(query: str) -> str:
    """查询地铁相关规章制度、乘车守则等官方文档。"""
    
    # ✅ 修正：通过模块名动态调用。这样测试里的 patch 才能拦截到
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
            results.append(f"【条款 {i+1}】: {doc.page_content.replace('\\n', ' ')}")
            
        return "\n\n".join(results)
    except Exception as e:
        return f"系统错误：知识库检索失败 ({str(e)})。"

# ...（下方 general_chat 函数中涉及 update_task_result 的地方，也请确保使用的是 utils.update_task_result 或正确导入）