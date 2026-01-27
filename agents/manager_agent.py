from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import agentState
from utils import complete_current_task, llm


async def manager_agent(state: agentState):
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "manager_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
    sys_msg = SystemMessage(content="你是管理助手，你的职责是协助完成质检、排班、监控培训相关工作。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    updated_board = complete_current_task(state, "manager_agent")
    return {
        "messages": [AIMessage(content=response.content, name="manager_agent")],
        "task_board": updated_board,
        "task_results": {"manager_agent": response.content}
    }