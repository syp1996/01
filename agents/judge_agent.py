from typing import Annotated, List, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from state import agentState
from utils import complete_current_task, llm

# --- 1. 定义搜索工具 ---
# DuckDuckGo 是免费的，适合开发测试。生产环境建议换成 Tavily 或 Google Serper
search_tool = DuckDuckGoSearchRun()

# 定义工具列表
tools = [search_tool]

# --- 2. 构建 ReAct 子图 ---

class JudgeAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm_with_tools = llm.bind_tools(tools)

def call_model(state: JudgeAgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
judge_workflow = StateGraph(JudgeAgentState)
judge_workflow.add_node("agent", call_model)
judge_workflow.add_node("tools", ToolNode(tools))

judge_workflow.add_edge(START, "agent")
judge_workflow.add_conditional_edges("agent", tools_condition)
judge_workflow.add_edge("tools", "agent")

judge_app = judge_workflow.compile()


# --- 3. 主函数 ---

async def judge_agent(state: agentState):
    # 1. 提取任务
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "judge_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
            
    if not isolated_input:
        return {"task_board": board}

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

    # 4. 销账
    updated_board = complete_current_task(state, "judge_agent")

    return {
        "messages": [AIMessage(content=final_content, name="judge_agent")],
        "task_board": updated_board,
        "task_results": {"judge_agent": final_content}
    }