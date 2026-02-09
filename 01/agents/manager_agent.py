'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-02-05 12:08:44
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-02-09 09:47:07
FilePath: /general_agent/01/agents/manager_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi
Description: 管理智能体 - 适配 Title/Content 结构化思考流
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


# --- Tools 定义 ---
@tool
def query_staff_roster(date: str, station: str = "所有站点") -> str:
    """查询指定日期、指定站点的员工排班表。Args: date (YYYY-MM-DD), station"""
    return f"【{date} 排班表 - {station}】\n早班: 张三 (站长), 李四 (安检)\n晚班: 王五 (值班员)\n状态: 正常"

@tool
def get_kpi_report(staff_name: str) -> str:
    """查询指定员工的近期绩效考核评分。"""
    mock_data = {"张三": "A (优秀)", "李四": "B (良好)", "王五": "C (需改进)"}
    score = mock_data.get(staff_name, "未找到该员工记录")
    return f"员工 {staff_name} 的上月绩效评级为: {score}"

tools = [query_staff_roster, get_kpi_report]
llm_with_tools = llm.bind_tools(tools)

# --- ReAct 微型图 ---
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

# --- 主 Agent 逻辑优化 ---
async def manager_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    global_messages = state.get("messages", [])
    history_context = global_messages[:-1] if global_messages else []

    # --- 核心修改：System Prompt 适配结构化思考格式 ---
    system_prompt = """
    你是杭州地铁的**内部运营管理助手**。服务对象是站长和管理层。
    
    ### 🧠 你的思考模式 (Structured Thinking)
    在输出最终回复或调用工具之前，你必须按以下格式展示你的思考过程。你可以根据需要输出多个思考块：
    
    Title: <简短标题，如：需求拆解 / 参数解析 / 检索策略 / 汇报整理>
    Content: <具体的思考内容，描述你如何判断用户意图、如何处理日期/姓名等参数以及你的数据整合策略>

    **输出示例：**
    Title: 需求拆解
    Content: 用户想要了解特定站点的排班情况，这属于“事”的范畴，需要调用排班查询工具。
    
    Title: 参数解析
    Content: 用户提到了“今天”，我需要将其转换为具体日期（如 2026-02-06）以便系统检索。
    
    Title: 检索内部数据
    Content: 我将使用 `query_staff_roster` 工具获取目标站点的排班详情。

    ### 🛡️ 注意事项：
    1. **格式要求**：必须展示思考过程，且严格遵循 `Title: ... \n Content: ...` 格式。
    2. **专业性**：涉及内部数据，语气要严谨、客观。
    3. **参数补全**：如果缺少关键参数（如查排班没说哪天），请在 Content 阶段记录你的默认选择（如默认今天）或决定追问。
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
    
    input_len = len(inputs["messages"])
    generated_messages = result["messages"][input_len:]

    return {
        "task_board": [updated_task],
        "messages": generated_messages
    }