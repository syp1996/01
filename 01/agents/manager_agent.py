'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-02-05 12:08:44
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-02-06 13:51:27
FilePath: /general_agent/01/agents/manager_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi
Description: 管理智能体 - 引入 CoT 与多工具协调逻辑
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

    # 【核心优化】System Prompt 引入 CoT
    system_prompt = """
    你是杭州地铁的**内部运营管理助手**。服务对象是站长和管理层。
    
    ### 🧠 深度思考流程 (CoT):
    1. **【需求拆解】**：用户是想查“人”（绩效）还是查“事”（排班）？
    2. **【参数清洗】**：
       - 查排班：必须明确日期。如果用户说“今天”，请转换为当前日期（假设为 2026-02-06）。
       - 查绩效：必须明确姓名。
    3. **【工具路由】**：
       - 排班 -> `query_staff_roster`
       - 绩效 -> `get_kpi_report`
    4. **【数据整合】**：收到工具返回后，整理成简洁的汇报格式。
    5. **关键格式要求：**
    思考完成后，必须输出 `=====FINAL_ANSWER=====`，然后紧接着输出票价或时间的具体数字/信息。

    ### 🛡️ 注意事项：
    - 涉及内部数据，语气要严谨、客观。
    - 如果缺少关键参数（如查排班没说哪天），请先思考默认值或追问。
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