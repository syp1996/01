'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/supervisor.py
Description: 总调度智能体 - 优化Prompt与分发逻辑，移除冗余标题生成
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
    核心调度节点：分析用户意图，生成任务看板 (Task Board)
    """
    current_board = state.get("task_board", [])
    updates = {}

    # 1. 【移除】旧的标题生成逻辑
    # 原因：已在 main.py 中实现了基于 LLM 的智能标题生成与 SSE 推送，此处不再需要。
    
    # 2. 仅当看板为空时（新一轮对话开始），进行规划
    if not current_board:
        # 动态获取当前时间，辅助决策（例如：用户问“现在有车吗”）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 格式化工人描述，让 LLM 更清楚每个人的职责
        members_desc = "\n".join([f"- **{k}**: {v}" for k, v in WORKERS_INFO.items()])
        
        system_prompt = f"""
        你是杭州地铁智能客服系统的**总调度官 (Supervisor)**。
        当前系统时间：{current_time}
        
        ### 你的职责：
        分析用户的输入，将其拆解为 1 个或多个具体的子任务，并分配给最合适的部门处理。
        
        ### 可选处理部门及其职责：
        {members_desc}
        
        ### 决策原则：
        1. **精准分发**：
           - 问票价/时刻/站点 -> 分配给 `ticket_agent`
           - 投诉/建议/反馈 -> 分配给 `complaint_agent`
           - 查排班/内部管理/绩效 -> 分配给 `manager_agent`
           - 问新闻/舆情/突发事件 -> 分配给 `judge_agent`
           - **闲聊/规章制度/无法归类的通用问题** -> 必须分配给 `general_chat`
           
        2. **参数提取**：
           - 尽可能从用户输入中提取关键信息作为 `input_content`。
           - 例如：用户说“查一下杭州东到武林广场的票价”，给 `ticket_agent` 的任务内容应保留完整意图。
           
        3. **多任务处理**：
           - 如果用户一句话包含多个意图（如“我要投诉安检，顺便问下末班车”），请拆分为两个独立的任务并行执行。
        """
        
        # 使用 Structured Output 强制生成规范的任务列表
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
                # 兜底：如果 LLM 未生成任务（极少情况），默认给 general_chat
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
    
    # 如果看板不为空（任务正在进行中或已完成），不更新看板，直接返回
    return updates

def workflow_router(state: agentState) -> Literal["responder_agent"] | List[Send]:
    """
    路由逻辑：
    - 检查 task_board 中状态为 'pending' 的任务。
    - 如果有，利用 LangGraph 的 Send 机制并行分发给对应的 Worker 节点。
    - 如果全完成了，转交给 responder_agent 进行汇总。
    """
    board = state.get("task_board", [])
    
    # 筛选出待处理的任务
    pending_tasks = [t for t in board if t["status"] == "pending"]
    
    if not pending_tasks:
        # 所有任务已结束 -> 汇总回复
        return "responder_agent"
    
    # 并行分发
    # 注意：我们将 'task' 对象和全局 'messages' 同时传给子智能体
    return [
        Send(node=task["task_type"], arg={"task": task, "messages": state["messages"]}) 
        for task in pending_tasks
    ]