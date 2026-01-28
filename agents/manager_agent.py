'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:57:25
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-28 13:40:21
FilePath: /01/agents/manager_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import agentState
from utils import complete_current_task, llm


async def manager_agent(state: agentState):
    # 1. 凭令牌取任务
    current_id = state.get("current_task_id")
    board = state.get("task_board", [])
    
    target_task = None
    for task in board:
        if task['id'] == current_id:
            target_task = task
            break

    if not target_task:
         return {"task_board": board}

    isolated_input = target_task['input_content']

    # 2. 执行逻辑
    sys_msg = SystemMessage(content="你是管理助手，你的职责是协助完成质检、排班、监控培训相关工作。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    
    # 3. 销账 (传入结果)
    updated_board = complete_current_task(state, result=response.content)
    
    return {
        "messages": [AIMessage(content=response.content, name="manager_agent")],
        "task_board": updated_board
    }