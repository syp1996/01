from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import agentState
from utils import complete_current_task, llm


async def judge_agent(state: agentState):
    # 1. 从看板获取【纯净输入】
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "judge_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
    sys_msg = SystemMessage(content="你是社情分析判断专家。只回和处理社情分析相关的问题，不要管其他问题。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    # 销账
    updated_board = complete_current_task(state, "judge_agent")
    return {
        "messages": [AIMessage(content=response.content, name="judge_agent")],
        "task_board": updated_board,# 返回更新后的看板
        # 3. 【新增】将结果写入暂存区
        "task_results": {"judge_agent": response.content}
    }
