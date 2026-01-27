from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import agentState
from utils import complete_current_task, llm


async def complaint_agent(state: agentState):
    sys_msg = SystemMessage(content="你是投诉专员。只回答投诉问题，不要管票务。")
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "complaint_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break

    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    updated_board = complete_current_task(state, "complaint_agent")
    
    return {
        "messages": [AIMessage(content=response.content, name="complaint_agent")],
        "task_board": updated_board,
        "task_results": {"complaint_agent": response.content}
    }