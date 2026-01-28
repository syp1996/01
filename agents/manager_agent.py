'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/manager_agent.py
Description: 并行化改造版
'''
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import WorkerState
from utils import llm, update_task_result


async def manager_agent(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']

    print(f"[Manager] 正在处理: {isolated_input}")

    # 保持你原有的 Manager Prompt
    sys_msg = SystemMessage(content="你是管理助手，你的职责是协助完成质检、排班、监控培训相关工作。")
    
    response = await llm.ainvoke([sys_msg, HumanMessage(content=isolated_input)])
    
    updated_task = update_task_result(task, result=response.content)
    
    return {
        "messages": [AIMessage(content=response.content, name="manager_agent")],
        "task_board": [updated_task]
    }