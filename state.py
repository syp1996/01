'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:55:06
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-28 13:29:39
FilePath: /01/state.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# 定义字典合并函数
def merge_dict(old_dict, new_dict):
    if not old_dict:
        return new_dict
    return {**old_dict, **new_dict}

# 单个任务模型
class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务唯一ID") # 自动生成UUID
    task_type: Literal["ticket_agent", "complaint_agent", "general_chat", "manager_agent", "judge_agent"]
    description: str = Field(..., description="任务的具体描述")
    input_content: str = Field(..., description="提取出的纯净输入")
    status: Literal["pending", "done"] = "pending"
    result: Optional[str] = None  # 直接把结果存在任务对象里，比放在全局字典更聚合

# 规划结果模型
class PlanningResponse(BaseModel):
    tasks: List[Task] = Field(..., description="任务列表")

# 全局状态定义
class agentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: str
    task_board: List[Dict[str, Any]]
    current_task_id: str # 新增：当前正在执行的任务指针    current_task_id: str # 新增：当前正在执行的任务指针