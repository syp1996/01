'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/general_chat.py
Description: 并行化改造版
'''
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import WorkerState
from utils import llm, update_task_result


async def general_chat(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']
    
    print(f"[General] 正在处理: {isolated_input}")

    sys_msg = SystemMessage(content="""
    你是地铁智能助手。
    负责回答用户的日常问候、闲聊以及非专业性的通用问题。
    请保持语气亲切、活泼。
    """)
    
    response = await llm.ainvoke([sys_msg, HumanMessage(content=isolated_input)])
    
    updated_task = update_task_result(task, result=response.content)
    
    return {
        "messages": [AIMessage(content=response.content, name="general_chat")],
        "task_board": [updated_task]
    }