'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/complaint_agent.py
Description: 并行化改造版
'''
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import WorkerState  # 【修改】使用 WorkerState
from utils import llm, update_task_result  # 【修改】使用 update_task_result


async def complaint_agent(state: WorkerState):
    # 1. 直接获取任务 (无需遍历)
    task = state["task"]
    isolated_input = task['input_content']
    
    print(f"[Complaint] 正在处理: {isolated_input}")

    # 2. 执行逻辑 (保持原有 Prompt)
    sys_msg = SystemMessage(content="""
    你是客户投诉处理专员。
    请安抚用户的情绪，记录投诉内容，并承诺会尽快反馈。
    请语气诚恳、专业、富有同理心。
    如果涉及具体站点或人员，请记录下来。
    """)
    
    response = await llm.ainvoke([sys_msg, HumanMessage(content=isolated_input)])
    
    # 3. 销账 (生成更新后的任务对象)
    updated_task = update_task_result(task, result=response.content)
    
    # 4. 返回结果
    return {
        "messages": [AIMessage(content=response.content, name="complaint_agent")],
        "task_board": [updated_task] # 列表形式返回，会自动合并
    }