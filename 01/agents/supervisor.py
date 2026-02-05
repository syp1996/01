'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/supervisor.py
Description: 增加自动提取标题逻辑
'''
import uuid
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from state import PlanningResponse, agentState
from utils import WORKERS_INFO, llm


async def supervisor_node(state: agentState):
    current_board = state.get("task_board", [])
    updates = {}

    # ✅ 新增逻辑：如果当前没有标题，取第一次提问的内容作为标题
    if not state.get("title"):
        user_input = ""
        # 寻找第一个 HumanMessage 作为标题来源
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break
        
        if user_input:
            # 截取前 20 个字符
            new_title = user_input[:20] + ("..." if len(user_input) > 20 else "")
            updates["title"] = new_title

    # 只有当看板完全为空时，才重新规划
    if not current_board:
        members_desc = "\n".join([f"- {k}: {v}" for k, v in WORKERS_INFO.items()])
        system_prompt = f"""
        你是一个智能客服系统的**任务规划师**。分析用户的输入并拆解任务。
        可选处理部门：{members_desc}
        """
        planner_chain = llm.with_structured_output(PlanningResponse, method="function_calling")
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        plan = await planner_chain.ainvoke(messages)

        new_board = []
        for task in plan.tasks:
            task_dict = task.model_dump()
            if not task_dict.get("id"):
                task_dict["id"] = str(uuid.uuid4())
            new_board.append(task_dict)
        
        updates["task_board"] = new_board
        return updates
    
    return updates

def workflow_router(state: agentState) -> Literal["responder_agent"] | List[Send]:
    board = state.get("task_board", [])
    pending_tasks = [t for t in board if t["status"] == "pending"]
    if not pending_tasks:
        return "responder_agent"
    return [Send(node=task["task_type"], arg={"task": task, "messages": state["messages"]}) for task in pending_tasks]