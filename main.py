'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-26 08:49:23
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-27 10:43:39
FilePath: /01/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import asyncio
import json
import os
import random
from operator import add  # 用于列表合并（LangGraph默认行为）
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

# 定义字典合并函数：将新结果合并到旧字典中，而不是覆盖
def merge_dict(old_dict, new_dict):
    if not old_dict:
        return new_dict
    return {**old_dict, **new_dict}

# 单个任务的模型
class Task(BaseModel):
    task_type: Literal["ticket_agent", "complaint_agent", "general_chat", "manager_agent", "judge_agent"]
    description: str = Field(..., description="任务的具体描述，例如：'查询西湖票价'")
    input_content: str = Field(..., description="用户关于该任务的具体输入内容。例如：'查询去萧山机场的路线'")
    status: Literal["pending", "done"] = "pending"

# 规划结果（监督者生成这个）
class PlanningResponse(BaseModel):
    tasks: List[Task] = Field(..., description="根据用户输入拆解出的任务列表")


# 初始化模型
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    max_tokens=4096
)

# 创建全局变量池
class agentState(TypedDict):
    messages:Annotated[List[BaseMessage], add_messages]
    next_step: str   # <--- 新增这个字段，用来存路由结果
    task_board: List[Dict[str, Any]] # 任务看板
    # 【新增】结果收集池
    # Annotated + merge_dict 意味着：当不同节点写入这个字段时，数据会合并
    # 例如：Ticket 写了 {ticket: "5元"}, Complaint 写了 {complaint: "已受理"}
    # 最终结果是两个都有。
    task_results: Annotated[Dict[str, str], merge_dict]


# 定义子节点能力描述 prompt
WORKERS_INFO = {
    "ticket_agent": "处理票务查询、票价计算、线路查询、首末班车时间。",
    "complaint_agent": "处理用户投诉、服务建议、设施故障反馈、失物招领。",
    "general_chat": "处理打招呼、问候、闲聊或无法归类的通用问题。",
    "manager_agent":"处理质检、排班、监控培训相关工作",
    "judge_agent":"负责热点分析,苗头事件,时序预测,线索筛查相关工作",
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

async def judge_agent(state: agentState):
    # 1. 从看板获取【纯净输入】
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "judge_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
    sys_msg = SystemMessage(content="你是社情分析判断专家。只回和处理社情分析相关的问题，不要管其他问题。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    # 销账
    updated_board = complete_current_task(state, "judge_agent")
    return {
        "messages": [AIMessage(content=response.content, name="judge_agent")],
        "task_board": updated_board,# 返回更新后的看板
        # 3. 【新增】将结果写入暂存区
        "task_results": {"judge_agent": response.content}
    }


async def ticket_agent(state: agentState):
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
        "task_board": updated_board,# 返回更新后的看板
        # 3. 【新增】将结果写入暂存区
        "task_results": {"ticket_agent": response.content}
    }

async def complaint_agent(state: agentState):
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
        "task_board": updated_board,
        "task_results": {"complaint_agent": response.content}
    }

async def manager_agent(state: agentState):
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "manager_agent" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
    sys_msg = SystemMessage(content="你是管理助手，你的职责是协助完成质检、排班、监控培训相关工作。")
    human_msg = HumanMessage(content=isolated_input)
    response = await llm.ainvoke([sys_msg, human_msg])
    updated_board = complete_current_task(state, "manager_agent")
    return {
        "messages": [AIMessage(content=response.content, name="manager_agent")],
        "task_board": updated_board,
        "task_results": {"manager_agent": response.content}
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
        "task_board": updated_board,
        "task_results": {"general_chat": response.content}
    }

async def responder_node(state: agentState):
    print("   [Responder] 所有任务已完成，正在生成最终回复...")
    
    # 1. 获取所有子智能体的劳动成果
    results = state.get("task_results", {})
    
    # 2. 获取用户的原始问题（用于给 LLM 提供上下文）
    # 我们倒序查找最后一条用户消息
    original_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            original_input = msg.content
            break

    # 3. 拼接上下文给 LLM
    context_str = "\n".join([f"【{k}的处理结果】: {v}" for k, v in results.items()])
    
    # 4. 编写 Prompt
    prompt = f"""
    你是一名专业的地铁客服经理。
    
    用户的原始问题是："{original_input}"
    
    你的各部门同事已经给出了处理结果，请你汇总这些信息，给用户一个**连贯、亲切、结构清晰**的最终回复。
    
    【各部门处理结果】：
    {context_str}
    
    【要求】：
    1. 不要暴露内部的 Agent 名称（如 ticket_agent）。
    2. 将零散的信息整合成一段通顺的话。
    3. 语气要统一，态度要专业且热情。
    """
    
    # 5. 调用模型生成最终回复
    final_response = await llm.ainvoke([SystemMessage(content=prompt)])
    
    # 6. 返回结果，并顺便清空看板和结果池（为下一轮对话重置状态）
    return {
        "messages": [AIMessage(content=final_response.content, name="final_responder")],
        "task_board": [],     # 清空看板
        "task_results": {}    # 清空结果池
    }

#构建图
workflow = StateGraph(agentState)

#添加节点
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("ticket_agent", ticket_agent)
workflow.add_node("complaint_agent", complaint_agent)
workflow.add_node("general_chat", chat_node)
workflow.add_node("manager_agent", manager_agent)
workflow.add_node("judge_agent", judge_agent)
workflow.add_node("responder_node", responder_node)

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
        "manager_agent": "manager_agent",
        "judge_agent": "judge_agent",
        "FINISH": "responder_node",  # 只有这里可以走向结束
    }
)

# 4. 设置出口：子节点干完活就结束 (END)
workflow.add_edge("ticket_agent", 'supervisor_node')
workflow.add_edge("complaint_agent", 'supervisor_node')
workflow.add_edge("general_chat", 'supervisor_node')
workflow.add_edge("manager_agent", 'supervisor_node')
workflow.add_edge("judge_agent", 'supervisor_node')
workflow.add_edge("responder_node", END)
# ---  初始化内存 ---
memory = MemorySaver()
# 编译
app = workflow.compile(checkpointer=memory)


# 运行测试函数
async def main():
    print(">>> 统一汇总模式已启动。")
    thread_id = "final_resp_test_002"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit"]: break
        
        input_state = {"messages": [HumanMessage(content=user_input)]}

        async for event in app.astream_events(input_state, config=config, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                
                # 【改动】只打印 responder 的输出
                if node_name == "responder_node":
                    chunk = event["data"]["chunk"]
                    if chunk.content: print(chunk.content, end="", flush=True)
                    
        print()

if __name__ == "__main__":
    asyncio.run(main())