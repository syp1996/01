'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/supervisor.py
Description: 总调度智能体 - 引入结构化思考 (Title/Content) 与任务分发逻辑
'''
import uuid
from datetime import datetime
from typing import List, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from state import PlanningResponse, agentState
from utils import WORKERS_INFO, llm


async def supervisor_node(state: agentState):
    """
    核心调度节点：分析用户意图，展示结构化思考过程，并生成任务看板 (Task Board)
    """
    current_board = state.get("task_board", [])
    updates = {}

    # 仅当看板为空时（新一轮对话开始），进行规划
    if not current_board:
        # 动态获取当前时间，辅助决策
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 格式化工人描述，让 LLM 清楚每个部门的职责
        members_desc = "\n".join([f"- **{k}**: {v}" for k, v in WORKERS_INFO.items()])
        
        # --- 核心修改：在 System Prompt 中加入结构化思考格式指令 ---
        system_prompt = f"""
        你是全能型的**私人智能助理总调度官 (Supervisor)**,你的名字叫贾维斯。
        当前系统时间：{current_time}
        
        ### 🧠 你的思考模式 (Structured Thinking)
        在生成最终的任务分配方案前，你必须向用户展示你的逻辑推演过程。
        **为了让前端正确展示你的思考步骤，请严格遵守以下格式输出（保留 Title/Content 标签）：**
        
        Title: <简短标题，例如：需求分析 / 工具选择策略 / 任务优先级判断>
        Content: <具体的思考内容，描述你如何理解用户需求，以及为何选择特定的助手处理>
        
        **输出示例 1 (查询类)：**
        Title: 需求分析
        Content: 用户想要了解“最新的苹果发布会内容”，这是一个需要获取实时信息的请求。
        Title: 任务分发
        Content: 本地知识库可能过时，我需要调度 `search_agent` 进行联网检索，然后由 `responder_agent` 总结。

        **输出示例 2 (复杂任务)：**
        Title: 意图拆解
        Content: 用户说“帮我查一下明天北京天气，并写一首关于雨天的诗”，这包含两个独立意图。
        Title: 策略制定
        Content: 任务1（查天气）分配给 `tools_agent` 或 `weather_agent`；任务2（写诗）分配给 `creative_writer`。两者可以并行。

        ### 你的职责：
        分析用户的输入，将其拆解为 1 个或多个具体的子任务，并分配给最合适的子智能体处理。
        
        ### 可选处理子智能体及其职责：
        {members_desc}
        
        ### 决策原则 (通用版)：
        1. **精准分发**：
           - **实时信息/百科/新闻/天气** -> 分配给 `search_agent` (或你定义的联网搜索工具)
           - **日程/提醒/邮件/日历操作** -> 分配给 `productivity_agent` (或工具类Agent)
           - **代码编写/数据分析/数学计算** -> 分配给 `code_interpreter` (或代码Agent)
           - **创意写作/文案润色/翻译** -> 分配给 `writer_agent` (或写作Agent)
           - **闲聊/情感陪伴/哲学探讨/无法归类的通用问题** -> 必须分配给 `general_chat`
           
        2. **参数提取**：
           - 尽可能从用户输入中提取关键信息（如地点、时间、主题）作为 `input_content`。
           - 如果用户询问“明天”，请结合当前时间 {current_time} 计算出具体日期。
           
        3. **多任务处理**：
           - 如果用户输入包含多个意图（例如：“查一下股价然后发邮件给老板”），请拆分为多个独立的任务。
        """
        
        # 使用 Structured Output 强制生成规范的任务列表
        # main.py 的流式解析器会捕捉 invoke 过程中产生的文本流（思考过程）
        planner_chain = llm.with_structured_output(PlanningResponse, method="function_calling")
        
        # 将 System Prompt 和 历史对话 传入
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        try:
            plan = await planner_chain.ainvoke(messages)
            
            new_board = []
            if plan and plan.tasks:
                for task in plan.tasks:
                    task_dict = task.model_dump()
                    # 确保每个任务都有唯一 ID
                    if not task_dict.get("id"):
                        task_dict["id"] = str(uuid.uuid4())
                    # 默认状态为 pending
                    task_dict["status"] = "pending"
                    new_board.append(task_dict)
            else:
                # 兜底逻辑
                new_board.append({
                    "id": str(uuid.uuid4()),
                    "task_type": "general_chat",
                    "input_content": state["messages"][-1].content,
                    "status": "pending"
                })
            
            updates["task_board"] = new_board
            
        except Exception as e:
            # 容错处理
            print(f"[Supervisor] Planning Error: {e}")
            updates["task_board"] = [{
                "id": str(uuid.uuid4()),
                "task_type": "general_chat",
                "input_content": state["messages"][-1].content,
                "status": "pending"
            }]

        return updates
    
    return updates

def workflow_router(state: agentState) -> Literal["responder_agent"] | List[Send]:
    """
    路由逻辑：
    - 检查 task_board 中状态为 'pending' 的任务并分发。
    """
    board = state.get("task_board", [])
    pending_tasks = [t for t in board if t["status"] == "pending"]
    
    if not pending_tasks:
        # 所有任务已结束 -> 汇总回复
        return "responder_agent"
    
    # 并行分发
    return [
        Send(node=task["task_type"], arg={"task": task, "messages": state["messages"]}) 
        for task in pending_tasks
    ]