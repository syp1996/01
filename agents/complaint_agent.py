'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:57:25
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-28 13:45:14
FilePath: /01/agents/complaint_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import agentState
from utils import complete_current_task, llm


async def complaint_agent(state: agentState):
    # 1. 凭令牌取任务 (ID Routing)
    current_id = state.get("current_task_id")
    board = state.get("task_board", [])
    
    target_task = None
    for task in board:
        if task['id'] == current_id:
            target_task = task
            break

    # 防御性编程：没找到任务直接返回
    if not target_task:
        return {"task_board": board}

    isolated_input = target_task['input_content']

    # 2. 执行核心逻辑
    sys_msg = SystemMessage(content="你是投诉专员。只回答投诉问题，不要管票务。")
    human_msg = HumanMessage(content=isolated_input)
    
    response = await llm.ainvoke([sys_msg, human_msg])
    
    # 3. 销账并保存结果
    updated_board = complete_current_task(state, result=response.content)
    
    return {
        "messages": [AIMessage(content=response.content, name="complaint_agent")],
        "task_board": updated_board
        # "task_results": ... <-- 已移除
    }