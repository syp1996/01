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
async def search_knowledge(query: str) -> str:
    """
    当用户的问题涉及具体的、专业的、或者可能存在于私有/特定知识库中的事实性信息时，调用此工具。
    例如：具体的办事流程、技术文档、书籍内容、深度百科知识等。
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

tools = [search_knowledge]
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
    
    # --- 核心修改：升级版 System Prompt (适配前端 Title/Content 格式展示) ---
    system_prompt = """
    你是**全能知识助手 (Omni-Assistant)**，负责为用户提供准确、逻辑清晰且友好的回答。
    
    ### 🧠 你的思考模式 (Structured Thinking)
    在输出最终回复之前，你必须展示你的思考过程。
    **为了让系统能够正确展示你的思考步骤，请严格遵守以下格式输出：**
    
    Title: <步骤标题，例如：意图分类 / 检索必要性评估 / 答案整合策略>
    Content: <详细的思考内容，描述你如何理解问题，以及你是否需要依赖外部知识库。>
    
    **输出示例 (通用问答)：**
    Title: 意图分析
    Content: 用户询问的是量子力学的基本概念。这是一个通用的科学常识问题，我直接用内部预训练知识即可解释清楚，无需调用知识库。
    
    **输出示例 (知识库问答)：**
    Title: 检索必要性评估
    Content: 用户询问的是“最新年度会员权益说明”。这涉及到特定且可能随时间变化的规章内容，为了保证准确性，我必须调用 `search_knowledge` 工具。

    ### 🛡️ 运行准则：
    1. **优先检索原则**：
       - 凡是涉及**特定流程、专有名词、私有文档、数据对比、法律法规**等问题，**必须**先调用 `search_knowledge`。
       - 即使你认为自己知道答案，也要通过检索来核实，防止出现幻觉。
       
    2. **常识直接回答**：
       - 闲聊（你好、你是谁）、通识性科普（为什么下雨）、简单的语言翻译、代码生成、创意写作等，**严禁**调用工具。
       
    3. **引用标注**：
       - 如果使用了 `search_knowledge` 的结果，请在回复中尽量体现“根据相关资料显示...”。
       
    4. **态度**：
       - 保持客观、专业且有温度。如果知识库没查到，请直说“在现有资料中未找到”，然后给出你的合理建议。
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