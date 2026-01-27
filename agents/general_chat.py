from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import agentState
from utils import complete_current_task, llm


async def general_chat(state: agentState):
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "general_chat" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
    sys_msg = SystemMessage(content="你是闲聊助手。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    
    updated_board = complete_current_task(state, "general_chat")
    
    return {
        "messages": [AIMessage(content=response.content, name="general_chat")],
        "task_board": updated_board,
        "task_results": {"general_chat": response.content}
    }