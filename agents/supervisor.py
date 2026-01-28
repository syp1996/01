'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:56:42
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-28 10:04:17
FilePath: /01/agents/supervisor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from langchain_core.messages import HumanMessage, SystemMessage

from state import PlanningResponse, agentState
from utils import WORKERS_INFO, llm


async def supervisor_node(state: agentState):
    # 从 state 获取当前看板，如果没有则初始化为空列表
    current_board = state.get("task_board", [])
    
    last_message = state["messages"][-1]
    # 判断是否是用户的新输入
    is_new_input = isinstance(last_message, HumanMessage)

    # --- 分支一：规划阶段 (Planning) ---
    # 只有当【用户刚说话】或者【看板完全为空】时，才重新规划
    if is_new_input or not current_board:
        print("\n[Supervisor] 检测到新需求，开始创建任务看板...")

        # 1. 构建 Prompt
        members_desc = "\n".join([f"- {k}: {v}" for k, v in WORKERS_INFO.items()])

        system_prompt = f"""
        你是一个智能客服系统的**任务规划师**。
        请分析用户的输入，将其拆解为一个个独立的子任务。
        
        可选的处理部门：
        {members_desc}
        
        **核心调度原则（请严格遵守）：**
        1. **General Chat (默认优先级)**: 凡是闲聊、问候、或者是询问普通的地铁政策、规定、建议等，**必须**派发给 `general_chat`。它是系统的默认兜底。
        2. **Complaint (严格限制)**: 只有当用户**明确表达了强烈的不满、愤怒**，或者**明确要求投诉工作人员/服务**时，才派发给 `complaint_agent`。
           - 例子："你们这什么服务态度！" -> complaint_agent
           - 例子："地铁里能吃东西吗？" -> general_chat (不要因为这是“质疑”就当成投诉)
        3. **Ticket**: 仅涉及具体的“票价查询”或“时刻表查询”时，派发给 `ticket_agent`。
        
        要求：
        1. 如果用户有多个意图，请拆分成多个任务。
        2. 必须输出 JSON 格式的任务列表。
        3. 对于每个子任务，你必须将用户原话中**属于该任务的部分**提取出来，填入 `input_content`。
        不要让 `input_content` 包含其他任务的信息。实现信息的物理隔离。
        """

        # 2. 调用 Planner
        planner_chain = llm.with_structured_output(PlanningResponse, method="function_calling")
        # 规划时只看 System Prompt 和 用户的那句话（或者完整历史），这里简单起见用完整历史
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        plan = await planner_chain.ainvoke(messages)

        # 【关键修改】在存入看板前，确保每个任务都有 ID
        new_board = []
        for task in plan.tasks:
            task_dict = task.model_dump()
            if not task_dict.get("id"):
                task_dict["id"] = str(uuid.uuid4()) # 双重保险，补全ID
            new_board.append(task_dict)

        # 3. 更新看板
        # new_board = [task.model_dump() for task in plan.tasks]
        print(f"[看板创建完毕]: {new_board}")
        
        # 将新生成的看板赋值给 current_board，准备进入路由阶段
        current_board = new_board

    # --- 分支二：路由阶段 (Routing) ---
    # 这里的代码在 if 外面，无论是否经过规划，都要执行路由
    
    # 查找第一个未完成的任务
    pending_task = None
    
    pending_task = None
    # 查找第一个未完成的任务
    for task in current_board:
        if task['status'] == 'pending':
            pending_task = task
            break
            
    if pending_task:
        target = pending_task['task_type']
        t_id = pending_task['id']
        print(f"[Supervisor] 派发任务 ID: {t_id} -> {target}")
        
        return {
            "task_board": current_board,
            "next_step": target,
            "current_task_id": t_id # 【关键】将令牌传给 Agent
        }
    else:
        return {
            "task_board": current_board, 
            "next_step": "FINISH",
            "current_task_id": ""
        }       