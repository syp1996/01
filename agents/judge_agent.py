'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:57:25
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-28 13:40:12
FilePath: /01/agents/judge_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import Annotated, List, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from state import agentState
from utils import complete_current_task, llm

# --- 工具和子图定义保持不变 ---
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

class JudgeAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm_with_tools = llm.bind_tools(tools)

def call_model(state: JudgeAgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

judge_workflow = StateGraph(JudgeAgentState)
judge_workflow.add_node("agent", call_model)
judge_workflow.add_node("tools", ToolNode(tools))
judge_workflow.add_edge(START, "agent")
judge_workflow.add_conditional_edges("agent", tools_condition)
judge_workflow.add_edge("tools", "agent")
judge_app = judge_workflow.compile()


# --- 主函数 (核心修改) ---
async def judge_agent(state: agentState):
    # 1. 凭令牌取任务
    current_id = state.get("current_task_id")
    board = state.get("task_board", [])
    
    target_task = None
    for task in board:
        if task['id'] == current_id:
            target_task = task
            break
            
    if not target_task:
        return {"task_board": board}

    isolated_input = target_task['input_content']

    # 2. 准备 Prompt
    system_prompt = """
    你是地铁舆情与社情分析专家。
    你需要对用户的提问进行实时分析。
    
    你的核心能力是【联网搜索】：
    1. 当用户询问具体的“新闻”、“事件”、“故障原因”、“公众评价”时，**必须调用 duckduckgo_search 工具**获取最新信息。
    2. 分析搜索结果，给出专业的判断和建议。
    3. 如果没有搜到相关信息，请基于常识回答，并注明信息来源有限。
    """

    # 3. 运行子图
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=isolated_input)
        ]
    }
    
    result = await judge_app.ainvoke(inputs)
    final_content = result["messages"][-1].content

    # 4. 销账 (传入结果)
    updated_board = complete_current_task(state, result=final_content)

    return {
        "messages": [AIMessage(content=final_content, name="judge_agent")],
        "task_board": updated_board
    }