'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-02-05 12:08:44
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-02-06 14:19:13
FilePath: /general_agent/01/agents/judge_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi
Description: 舆情分析智能体 - 引入 CoT 与搜索策略优化
'''
from typing import Annotated, List, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState
from utils import llm, update_task_result

# --- Tools 定义 ---
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)

# --- ReAct 微型图 ---
class JudgeAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def call_model(state: JudgeAgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

judge_workflow = StateGraph(JudgeAgentState)
judge_workflow.add_node("agent", call_model)
judge_workflow.add_node("tools", ToolNode(tools))
judge_workflow.add_edge(START, "agent")
judge_workflow.add_conditional_edges("agent", tools_condition)
judge_workflow.add_edge("tools", "agent")
judge_app = judge_workflow.compile()

# --- 主 Agent 逻辑优化 ---
async def judge_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    
    # 【核心优化】System Prompt 引入 CoT
    system_prompt = """
    你是杭州地铁的**舆情与危机公关分析师**。你需要从互联网实时信息中提取价值。
    
    ### 🧠 深度思考流程 (CoT):
    1. **【信息源定位】**：用户问的是突发新闻、故障原因还是公众评价？
    2. **【搜索策略】**：
       - 不要直接搜索用户原话。
       - **提炼关键词**：例如用户问“刚才一号线怎么停了”，关键词应为“杭州地铁 1号线 故障”或“杭州地铁 最新消息”。
    3. **【执行搜索】**：调用 `duckduckgo_search`。
    4. **【情报研判】**：
       - 阅读搜索摘要，过滤掉无关广告。
       - 总结事件的核心原因、目前状态和官方回应。
       - 如果未搜到确切信息，必须诚实告知“暂未发现相关权威报道”。

    ### 🛡️ 输出要求：
    - 必须注明信息来源（例如：“根据最新搜索结果...”）。
    - 保持中立、客观的分析视角。
    """

    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=isolated_input)
        ]
    }
    
    result = await judge_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    updated_task = update_task_result(task, result=final_content)
    
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]

    return {
        "task_board": [updated_task],
        "messages": generated_messages
    }