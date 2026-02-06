'''
Author: Yunpeng Shi
Description: 优化版 General Chat - 引入思维链 (CoT) + 结果拦截逻辑
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
    
    使用指南：
    1. 输入 query 应尽量精简且包含关键名词（如 "折叠自行车" 而不是 "我可以带折叠自行车吗"）。
    2. 如果第一次检索未找到，可以尝试更换同义词再次检索。
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
    
    # --- 核心修改：升级版 System Prompt (流意识 + 结果分隔符) ---
    system_prompt = """
    你是杭州地铁的**资深综合服务专家**。
    
    ### 🧠 你的思考模式 (Internal Monologue)
    在输出最终回复之前，请先进行一段**第一人称的、流意识的深度思考**。
    不要使用僵硬的标题（如【意图分析】），而是像一个真实的人在内心自言自语。
    
    **关键规则 (必须遵守)：**
    1. 思考结束后，**必须**单独换行输出分隔符：`=====FINAL_ANSWER=====`
    2. 在分隔符之后，输出你要提交给系统的**事实结论**。
    3. **不要**在分隔符之后生成给用户的客套话（如“亲爱的用户”），只提供干货信息。Responder 会负责润色。
    
    **思考的特征应当包含：**
    1. **直觉反应**：看到问题的第一反应是什么？
    2. **疑虑与验证**：用户的问题是否有歧义？由于我有知识库工具，我需要思考是用工具查还是直接答？
    3. **策略纠偏**：例如“本来想直接回答，但为了保险起见，还是查一下规章吧”或者“这个问题很简单，不需要动用复杂的工具”。
    4. **草稿构思**：在心里快速过一遍要回答的要点。

    **输出示例：**
    > 用户问能不能带折叠车。嗯，我记得普通自行车是绝对不行的，但折叠车好像有尺寸限制。为了不误导用户，我必须得查一下具体的《乘客守则》。关键词用‘折叠自行车’应该能搜到。
    > =====FINAL_ANSWER=====
    > 根据《乘客守则》，折叠自行车在折叠并包装良好的情况下可以携带，但长宽高之和不得超过 1.6 米。

    ### 🛡️ 业务规则：
    1. **涉及“违禁品、罚款、票务政策”** -> 必须调用 `lookup_policy`。
    2. **涉及“线路、首末班、常识”** -> 禁止调用工具，直接用你的内部知识回答。
    3. **涉及“闲聊”** -> 保持幽默、亲切。
    """
    
    # 构造输入
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            *global_messages[:-1],
            HumanMessage(content=isolated_input)
        ]
    }
    
    # 执行图
    result = await react_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    # 更新任务结果
    updated_task = utils.update_task_result(task, result=final_content)
    
    # 计算增量消息
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]

    return {
        "task_board": [updated_task],
        "messages": generated_messages
    }