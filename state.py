'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:55:06
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/state.py
Description: 并行化改造版 - 引入 Reducer 和 WorkerState
'''
import uuid
from typing import (Annotated, Any, Dict, List, Literal, Optional, TypedDict,
                    Union)

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# --- Reducer 函数 (核心新增) ---
def reduce_task_board(left: List[Dict[str, Any]], right: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    任务看板的归并函数。
    当多个 Agent 并行返回结果时，该函数会确保所有人的更新都能正确合并到总看板中。
    
    Args:
        left: 当前的状态 (旧看板)
        right: 新的状态更新 (可以是单个任务字典，也可以是任务列表)
    """
    # 1. 规范化输入：确保 right 是列表
    updates = [right] if isinstance(right, dict) else right
    
    # 2. 如果旧看板为空（初始化阶段），直接返回新列表
    if not left:
        return updates
    
    # 3. 复制旧看板，准备进行合并
    new_board = left.copy()
    
    # 建立 ID 到 索引 的映射，由 O(N^2) 优化为 O(N)
    id_to_index = {task["id"]: i for i, task in enumerate(new_board)}
    
    for update in updates:
        t_id = update.get("id")
        if t_id and t_id in id_to_index:
            # 找到对应任务，进行“原地”更新 (合并字段)
            idx = id_to_index[t_id]
            # 使用字典解包合并：旧数据 + 新数据 (新覆盖旧)
            new_board[idx] = {**new_board[idx], **update}
        else:
            # 如果是没见过的任务 ID (理论上不应发生，作为容错)，则追加
            new_board.append(update)
            
    return new_board


# --- Pydantic 模型 ---
class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务唯一ID") 
    task_type: Literal["ticket_agent", "complaint_agent", "general_chat", "manager_agent", "judge_agent"]
    description: str = Field(..., description="任务的具体描述")
    input_content: str = Field(..., description="提取出的纯净输入")
    status: Literal["pending", "done"] = "pending"
    result: Optional[str] = None 

class PlanningResponse(BaseModel):
    tasks: List[Task] = Field(..., description="任务列表")


# --- Worker 专用状态 (核心新增) ---
class WorkerState(TypedDict):
    """
    这是传递给并行 Worker 的'私有'状态。
    Supervisor 使用 Send() 分发任务时，Payload 会匹配这个结构。
    """
    task: Dict[str, Any]


# --- 全局状态 ---
class agentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
    next_step: str
    
    # 【关键修改】使用 Annotated + Reducer 来管理看板
    # 这样允许多个 Worker 同时返回 {"task_board": 更新的任务} 而不冲突
    task_board: Annotated[List[Dict[str, Any]], reduce_task_board]
    
    # 已移除 current_task_id    # 已移除 current_task_id