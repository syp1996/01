'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-26 08:49:23
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-27 08:23:56
FilePath: /01/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import asyncio
import json
import os
import random
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# 单个任务的模型
class Task(BaseModel):
    task_type: Literal["ticket_agent", "complaint_agent", "general_chat"]
    description: str = Field(..., description="任务的具体描述，例如：'查询西湖票价'")
    input_content: str = Field(..., description="用户关于该任务的具体输入内容。例如：'查询去萧山机场的路线'")
    status: Literal["pending", "done"] = "pending"

# 规划结果（监督者生成这个）
class PlanningResponse(BaseModel):
    tasks: List[Task] = Field(..., description="根据用户输入拆解出的任务列表")

# 定义路由输出的结构
class RouteResponse(BaseModel):
    next_node:Literal["ticket_agent", "complaint_agent", "general_chat","FINISH"] = Field(
        ..., description="根据用户意图选择的下一个处理节点")

# 初始化模型
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
)

# 创建全局变量池
class agentState(TypedDict):
    messages:Annotated[List[BaseMessage], add_messages]
    next_step: str   # <--- 新增这个字段，用来存路由结果
    task_board: List[Dict[str, Any]]


# 定义子节点能力描述 prompt
WORKERS_INFO = {
    "ticket_agent": "处理票务查询、票价计算、线路查询、首末班车时间。",
    "complaint_agent": "处理用户投诉、服务建议、设施故障反馈、失物招领。",
    "general_chat": "处理打招呼、问候、闲聊或无法归类的通用问题。"
}

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

# ---  边逻辑 ---
def get_next_node(state: agentState) -> str:
    return state.get("next_step", "FINISH")

# 辅助函数：帮助子智能体“销账”
def complete_current_task(state: agentState, agent_name: str):
    board = state.get("task_board", [])
    # 找到属于我的第一个 pending 任务
    new_board = []
    marked = False
    
    for task in board:
        # 必须把 task 复制一份，否则可能是引用修改
        t = task.copy()
        if not marked and t['task_type'] == agent_name and t['status'] == 'pending':
            t['status'] = 'done' # 勾选！
            marked = True
            print(f"   >>> [{agent_name}] 完成任务，已在看板勾选。")
        new_board.append(t)
        
    return new_board

async def ticket_node(state: agentState):
    # 1. 从看板获取【纯净输入】
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "ticket_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
    sys_msg = SystemMessage(content="你是票务专家。只回答票务问题，不要管投诉。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    
    # 销账
    updated_board = complete_current_task(state, "ticket_agent")
    
    return {
        "messages": [AIMessage(content=response.content, name="ticket_agent")],
        "task_board": updated_board # 返回更新后的看板
    }

async def complaint_node(state: agentState):
    sys_msg = SystemMessage(content="你是投诉专员。只回答投诉问题，不要管票务。")
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "complaint_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break

    human_msg = HumanMessage(content=isolated_input)
    
    response = await llm.ainvoke([sys_msg, human_msg])
    
    updated_board = complete_current_task(state, "complaint_agent")
    
    return {
        "messages": [AIMessage(content=response.content, name="complaint_agent")],
        "task_board": updated_board
    }

async def chat_node(state: agentState):
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "general_chat" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
    sys_msg = SystemMessage(content="你是闲聊助手。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    
    updated_board = complete_current_task(state, "general_chat")
    
    return {
        "messages": [AIMessage(content=response.content, name="general_chat")],
        "task_board": updated_board
    }

#构建图
workflow = StateGraph(agentState)

#添加节点
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("ticket_agent", ticket_node)
workflow.add_node("complaint_agent", complaint_node)
workflow.add_node("general_chat", chat_node)


# 所有请求先进 supervisor
workflow.add_edge(START, 'supervisor_node')
# 3. 设置条件边：supervisor -> (根据逻辑) -> 具体节点
workflow.add_conditional_edges(
    "supervisor_node",      # 出发点
    get_next_node,     # 路由逻辑函数
    {                  # 路径映射表
        "ticket_agent": "ticket_agent",
        "complaint_agent": "complaint_agent",
        "general_chat": "general_chat",
        "FINISH": END  # 只有这里可以走向结束
    }
)

# 4. 设置出口：子节点干完活就结束 (END)
workflow.add_edge("ticket_agent", 'supervisor_node')
workflow.add_edge("complaint_agent", 'supervisor_node')
workflow.add_edge("general_chat", 'supervisor_node')

# ---  初始化内存 ---
memory = MemorySaver()
# 编译
app = workflow.compile(checkpointer=memory)


# 运行测试函数
async def main():
    print(">>> 任务看板模式已启动。请输入：'查一下去火车东站的票，另外投诉一下刚刚进站太慢了'")
    thread_id = "board_test_001"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit"]:
            break
            
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            # 每次新对话开始时，我们可以选择清空看板，
            # 或者由 Supervisor 的逻辑去判断覆盖。
            # 这里我们让 Supervisor 的逻辑去处理。
        }

        async for event in app.astream_events(input_state, config=config, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name == "supervisor_node":
                    continue
                
                chunk = event["data"]["chunk"]
                if chunk.content:
                    print(chunk.content, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())