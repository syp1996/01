'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:56:42
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/supervisor.py
Description: 并行化改造版 - 修复 Send 导入路径
'''
import uuid
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
# 【修正】Send 位于 langgraph.types，而非 langgraph.graph
from langgraph.types import Send

from state import PlanningResponse, agentState
from utils import WORKERS_INFO, llm


async def supervisor_node(state: agentState):
    """
    Supervisor 节点：仅负责任务规划（Planning），不再负责路由。
    它会检查是否需要生成新的任务看板，并更新到 State 中。
    """
    current_board = state.get("task_board", [])
    last_message = state["messages"][-1]
    is_new_input = isinstance(last_message, HumanMessage)

    # --- 规划逻辑 ---
    # 只有当【用户刚说话】或者【看板完全为空】时，才重新规划
    if is_new_input or not current_board:
        print("\n[Supervisor] 检测到新需求，开始创建任务看板...")

        members_desc = "\n".join([f"- {k}: {v}" for k, v in WORKERS_INFO.items()])
        
        system_prompt = f"""
        你是一个智能客服系统的**任务规划师**。
        请分析用户的输入，将其拆解为一个个独立的子任务。
        
        可选的处理部门：
        {members_desc}
        
        **核心调度原则（请严格遵守）：**
        1. **General Chat (默认优先级)**: 凡是闲聊、问候、或者是询问普通的地铁政策、规定、建议等，**必须**派发给 `general_chat`。
        2. **Complaint (严格限制)**: 只有当用户**明确表达了强烈的不满**时，才派发给 `complaint_agent`。
        3. **Ticket**: 仅涉及具体的“票价查询”或“时刻表查询”时，派发给 `ticket_agent`。
        
        要求：
        1. 必须输出 JSON 格式的任务列表。
        2. 对于每个子任务，提取出纯净的 `input_content`，实现信息的物理隔离。
        """

        planner_chain = llm.with_structured_output(PlanningResponse, method="function_calling")
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        # 调用大模型生成计划
        plan = await planner_chain.ainvoke(messages)

        # 补全 ID 并构建新看板
        new_board = []
        for task in plan.tasks:
            task_dict = task.model_dump()
            if not task_dict.get("id"):
                task_dict["id"] = str(uuid.uuid4())
            new_board.append(task_dict)

        print(f"[看板创建完毕]: {len(new_board)} 个任务 -> {[t['task_type'] for t in new_board]}")
        
        # 返回状态更新
        return {"task_board": new_board}
    
    # 如果不是新输入，不做任何规划
    return {}


def workflow_router(state: agentState) -> Literal["responder_agent"] | List[Send]:
    """
    【新增】条件边路由函数 (Conditional Edge)
    这是实现并行的核心。它检查看板，如果有多个 pending 任务，就返回多个 Send 对象。
    """
    board = state.get("task_board", [])
    
    # 1. 筛选出所有待处理的任务
    pending_tasks = [t for t in board if t["status"] == "pending"]
    
    # 2. 如果没有待办任务，说明全部完成，转交给 Responder 汇总回复
    if not pending_tasks:
        return "responder_agent"
    
    # 3. 并行分发！
    print(f"[Router] 并行分发 {len(pending_tasks)} 个任务...")
    
    return [
        Send(task["task_type"], {"task": task}) 
        for task in pending_tasks
    ]