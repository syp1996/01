'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-26 08:49:23
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/main.py
Description: 并行化改造版 - 支持 Map-Reduce 架构
'''
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# 导入所有智能体
# 注意：目前只有 ticket_agent 完成了并行化改造，其他 Agent 暂时不能使用
from agents.complaint_agent import complaint_agent
from agents.general_chat import general_chat
from agents.judge_agent import judge_agent
from agents.manager_agent import manager_agent
from agents.responder_agent import responder_agent
from agents.supervisor import supervisor_node, workflow_router  # 【关键】导入路由函数
from agents.ticket_agent import ticket_agent
from state import agentState

load_dotenv()

# --- 构建图 ---
workflow = StateGraph(agentState)

# 1. 添加节点
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("ticket_agent", ticket_agent)
# 其他节点暂时保留，但在全部改造完成前不要触发它们
workflow.add_node("complaint_agent", complaint_agent)
workflow.add_node("general_chat", general_chat)
workflow.add_node("manager_agent", manager_agent)
workflow.add_node("judge_agent", judge_agent)
workflow.add_node("responder_agent", responder_agent)

# 2. 设置连线
workflow.add_edge(START, 'supervisor_node')

# 3. 设置条件边 (核心变化)
# 不再使用 get_next_node，而是使用 workflow_router
# 它会返回 Send 对象列表（并行）或者 "responder_agent"（结束）
workflow.add_conditional_edges(
    "supervisor_node",
    workflow_router,
    # 这里的列表告诉 LangGraph 可能的去向，用于构建图结构
    [
        "ticket_agent", 
        "complaint_agent", 
        "general_chat", 
        "manager_agent", 
        "judge_agent", 
        "responder_agent"
    ]
)

# 4. 设置回环 (Worker -> Supervisor)
# 并行执行完后，必须回到 Supervisor 检查是否所有任务都完成了
workflow.add_edge("ticket_agent", "supervisor_node")
workflow.add_edge("complaint_agent", "supervisor_node")
workflow.add_edge("general_chat", "supervisor_node")
workflow.add_edge("manager_agent", "supervisor_node")
workflow.add_edge("judge_agent", "supervisor_node")

workflow.add_edge("responder_agent", END)

# --- 编译与运行 ---
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

async def main():
    print(">>> Map-Reduce 并行架构已启动 (测试模式)。")
    print(">>> ⚠️ 警告: 目前仅 Ticket Agent 已改造，请只测试票务问题。")
    
    thread_id = "parallel_test_001"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit"]: break
        
        input_state = {"messages": [HumanMessage(content=user_input)]}

        async for event in app.astream_events(input_state, config=config, version="v1"):
            kind = event["event"]
            
            # 打印流式输出，方便观察并行过程
            if kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name:
                    chunk = event["data"]["chunk"]
                    if chunk.content: 
                        #在此处加个前缀，看看是谁在说话
                        # print(f"[{node_name}]: {chunk.content}", end="", flush=True) 
                        print(chunk.content, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())