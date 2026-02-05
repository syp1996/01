'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/judge_agent.py
Description: 并行化改造版 - 包含子图结构
'''
from typing import Annotated, List, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState  # 修改引入
from utils import llm, update_task_result  # 修改引入

# --- 工具和子图定义 (完全保持不变) ---
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


# --- 主 Agent 函数 (核心修改) ---
async def judge_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    
    print(f"[Judge] 正在处理: {isolated_input}")

    # 保持原有的 Prompt
    system_prompt = """
    你是地铁舆情与社情分析专家。
    你需要对用户的提问进行实时分析。
    
    你的核心能力是【联网搜索】：
    1. 当用户询问具体的“新闻”、“事件”、“故障原因”、“公众评价”时，**必须调用 duckduckgo_search 工具**获取最新信息。
    2. 分析搜索结果，给出专业的判断和建议。
    3. 如果没有搜到相关信息，请基于常识回答，并注明信息来源有限。
    """

    # 运行子图
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=isolated_input)
        ]
    }
    
    result = await judge_app.ainvoke(inputs)
    final_content = result["messages"][-1].content

    # 销账
    updated_task = update_task_result(task, result=final_content)

    return {
        # 修复：移除 messages 返回，防止污染全局历史
        "task_board": [updated_task]
    }