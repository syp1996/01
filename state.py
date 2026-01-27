from typing import Annotated, Any, Dict, List, Literal, TypedDict

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
    task_type: Literal["ticket_agent", "complaint_agent", "general_chat", "manager_agent", "judge_agent"]
    description: str = Field(..., description="任务的具体描述")
    input_content: str = Field(..., description="提取出的纯净输入")
    status: Literal["pending", "done"] = "pending"

# 规划结果模型
class PlanningResponse(BaseModel):
    tasks: List[Task] = Field(..., description="任务列表")

# 全局状态定义
class agentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_step: str
    task_board: List[Dict[str, Any]]
    task_results: Annotated[Dict[str, str], merge_dict]