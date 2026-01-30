'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-28 15:19:55
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-30 16:34:13
FilePath: /general_agent/01/agents/general_chat.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi
Description: 修复 Mock 拦截逻辑的业务代码
'''
import os
from typing import Annotated, List, TypedDict

import utils  # ✅ 必须导入整个模块
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState

# ⚠️ 注意：这里不要用 from utils import get_vector_store

@tool
async def lookup_policy(query: str) -> str:
    """查询地铁相关规章制度、乘车守则等官方文档。"""
    
    # ✅ 关键修正：通过 utils 模块名动态调用
    # 这样 Mock 机器人就能在 utils 模块里精准拦截
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
        
        results = [f"【条款 {i+1}】: {doc.page_content.replace('\\n', ' ')}" for i, doc in enumerate(docs)]
        return "\n\n".join(results)
    except Exception as e:
        return f"系统错误：知识库检索失败 ({str(e)})。"

# ... (下方 rag_workflow 和 general_chat 保持不变，但确保 update_task_result 调用也带上 utils. 或者正确导入)