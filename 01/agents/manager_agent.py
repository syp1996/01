'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/manager_agent.py
Description: 并行化 ReAct 改造版 - 增加内部管理查询能力
'''
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from state import WorkerState
from utils import llm, update_task_result


# --- 1. 定义工具 (Tools) ---
@tool
def query_staff_roster(date: str, station: str = "所有站点") -> str:
    """
    查询指定日期、指定站点的员工排班表。
    Args:
        date: 日期 (YYYY-MM-DD)
        station: 站点名称，默认为"所有站点"
    """
    # 模拟数据
    return f"【{date} 排班表 - {station}】\n早班: 张三 (站长), 李四 (安检)\n晚班: 王五 (值班员)\n状态: 正常"

@tool
def get_kpi_report(staff_name: str) -> str:
    """
    查询指定员工的近期绩效考核评分。
    Args:
        staff_name: 员工姓名
    """
    mock_data = {"张三": "A (优秀)", "李四": "B (良好)", "王五": "C (需改进)"}
    score = mock_data.get(staff_name, "未找到该员工记录")
    return f"员工 {staff_name} 的上月绩效评级为: {score}"

tools = [query_staff_roster, get_kpi_report]

# --- 2. 构建 ReAct 子图 ---
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm_with_tools = llm.bind_tools(tools)

def call_model(state: SubAgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

worker_workflow = StateGraph(SubAgentState)
worker_workflow.add_node("model", call_model)
worker_workflow.add_node("tools", ToolNode(tools))

worker_workflow.add_edge(START, "model")
worker_workflow.add_conditional_edges("model", tools_condition)
worker_workflow.add_edge("tools", "model")

react_app = worker_workflow.compile()

# --- 3. 主 Agent 函数 ---
async def manager_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']

    print(f"[Manager] 正在处理 (ReAct): {isolated_input}")
    
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []

    # System Prompt：设定管理专家人设
    system_prompt = """
    你是杭州地铁的内部管理助手。
    你的服务对象是地铁工作人员和管理层。
    
    你可以帮助查询排班、考核绩效、或者提供管理建议。
    
    **工具使用原则：**
    1. 当用户问及人员去向、排班情况时，请使用 `query_staff_roster`。
    2. 当用户问及表现、考核时，请使用 `get_kpi_report`。
    3. 如果是通用的管理咨询（如：如何提升团队士气），直接依靠你的知识回答即可。
    """
    
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            *history_context,
            HumanMessage(content=isolated_input)
        ]
    }
    
    result = await react_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    updated_task = update_task_result(task, result=final_content)
    
    return {
        # 修复：移除 messages 返回，防止污染全局历史
        "task_board": [updated_task]
    }