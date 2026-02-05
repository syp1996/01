'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/state.py
Description: 增加 title 字段以支持对话命名
'''
import uuid
from typing import (Annotated, Any, Dict, List, Literal, Optional, TypedDict,
                    Union)

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# --- Reducer 函数 ---
def reduce_task_board(left: List[Dict[str, Any]], right: Union[List[Dict[str, Any]], Dict[str, Any], str]) -> List[Dict[str, Any]]:
    if right == "RESET":
        return []
    updates = [right] if isinstance(right, dict) else right
    if not left:
        return updates
    new_board = left.copy()
    id_to_index = {task["id"]: i for i, task in enumerate(new_board)}
    for update in updates:
        t_id = update.get("id")
        if t_id and t_id in id_to_index:
            idx = id_to_index[t_id]
            new_board[idx] = {**new_board[idx], **update}
        else:
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

# --- Worker 专用状态 ---
class WorkerState(TypedDict):
    task: Dict[str, Any]
    messages: List[BaseMessage]

# --- 全局状态 ---
class agentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: str
    task_board: Annotated[List[Dict[str, Any]], reduce_task_board]
    # ✅ 新增：存储当前对话的标题
    title: Optional[str]