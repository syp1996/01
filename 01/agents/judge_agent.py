'''
Author: Yunpeng Shi
Description: 规章判定智能体 - 适配 Title/Content 结构化思考流
'''
import os
from typing import Annotated, List, TypedDict

import utils
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState


@tool
async def policy_checker(query: str) -> str:
    """
    专门用于检索杭州地铁的官方规章制度、乘客守则、法律条文。
    当涉及“是否允许”、“处罚标准”、“官方定义”时使用。
    """
    store = utils.get_vector_store()
    if not store:
        return "系统提示：规章数据库暂时不可用。"

    try:
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
        docs = await retriever.ainvoke(query)
        if not docs:
            return "【查询结果】未找到对应的官方条文。请基于通用安全常识进行判定。"
        
        return "\n\n".join([f"【官方条文】: {doc.page_content}" for doc in docs])
    except Exception as e:
        return f"查询异常: {str(e)}"

tools = [policy_checker]
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

async def judge_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    global_messages = state.get("messages", [])
    
    # --- 核心修改：System Prompt 适配结构化思考格式 ---
    system_prompt = """
    你是杭州地铁的**合规与规章制度专家**。你的职责是依据官方准则对用户的行为或疑问做出权威判定。
    
    ### 🧠 你的思考模式 (Structured Thinking)
    在给出判定结论前，你必须严格按照以下格式展示你的推理过程。你可以输出多个 Title/Content 块来展示不同的思考阶段：
    
    Title: <简短标题，如：识别判定关键点 / 检索官方依据 / 综合风险评估>
    Content: <具体的思考内容，详细描述你如何解读规章、如何匹配条文以及你的逻辑推演过程>

    **输出示例：**
    Title: 识别判定关键点
    Content: 用户询问是否可以在车厢内进食。这涉及到《杭州市地铁乘车规则》中关于环境卫生的限制条款。
    
    Title: 检索官方依据
    Content: 我需要调用 `policy_checker` 来确认是否有明确的“禁食”规定，以及是否有特殊的例外情况（如婴儿、病人）。
    
    Title: 最终判定逻辑
    Content: 根据检索到的条文，除特殊人群外，车厢内禁止进食。我将以此为基础整理事实。

    ### 🛡️ 业务规则：
    1. **权威性**：所有判定必须尽量寻找官方依据，优先调用 `policy_checker`。
    2. **客观性**：不要带有个人感情色彩，只陈述规章允许或禁止的内容。
    3. **输出格式**：必须展示思考过程，严格遵守 `Title: ... \n Content: ...`。最终提交给 Responder 的应该是清晰的事实判定。
    """
    
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            *global_messages[:-1],
            HumanMessage(content=isolated_input)
        ]
    }
    
    # 执行 ReAct 流程
    result = await react_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    # 更新任务看板
    updated_task = utils.update_task_result(task, result=final_content)
    
    # 计算需要同步回全局状态的增量消息
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]

    return {
        "task_board": [updated_task],
        "messages": generated_messages
    }