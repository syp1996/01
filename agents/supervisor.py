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

        # 3. 更新看板
        new_board = [task.model_dump() for task in plan.tasks]
        print(f"[看板创建完毕]: {new_board}")
        
        # 将新生成的看板赋值给 current_board，准备进入路由阶段
        current_board = new_board

    # --- 分支二：路由阶段 (Routing) ---
    # 这里的代码在 if 外面，无论是否经过规划，都要执行路由
    
    # 查找第一个未完成的任务
    pending_task = None
    
    for i, task in enumerate(current_board):
        if task['status'] == 'pending':
            pending_task = task
            break
            
    if pending_task:
        target = pending_task['task_type']
        print(f"[Supervisor] 发现待办任务: {pending_task['description']} -> 派给 {target}")
        return {
            "task_board": current_board, # 确保把（可能新建的）看板存回 State
            "next_step": target
        }
    else:
        print("[Supervisor] 看板上所有任务已勾选 (Status: done)。结束流程。")
        return {
            "task_board": current_board, 
            "next_step": "FINISH"
        }        