'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-28 15:53:34
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-28 15:59:56
FilePath: /01/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Description: 并行化改造版 - 优化显示逻辑，解决乱码和重复输出问题
'''
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# 导入所有智能体
from agents.complaint_agent import complaint_agent
from agents.general_chat import general_chat
from agents.judge_agent import judge_agent
from agents.manager_agent import manager_agent
from agents.responder_agent import responder_agent
from agents.supervisor import supervisor_node, workflow_router
from agents.ticket_agent import ticket_agent
from state import agentState

load_dotenv()

# --- 构建图 (保持不变) ---
workflow = StateGraph(agentState)

workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("ticket_agent", ticket_agent)
workflow.add_node("complaint_agent", complaint_agent)
workflow.add_node("general_chat", general_chat)
workflow.add_node("manager_agent", manager_agent)
workflow.add_node("judge_agent", judge_agent)
workflow.add_node("responder_agent", responder_agent)

workflow.add_edge(START, 'supervisor_node')

workflow.add_conditional_edges(
    "supervisor_node",
    workflow_router,
    [
        "ticket_agent", 
        "complaint_agent", 
        "general_chat", 
        "manager_agent", 
        "judge_agent", 
        "responder_agent"
    ]
)

workflow.add_edge("ticket_agent", "supervisor_node")
workflow.add_edge("complaint_agent", "supervisor_node")
workflow.add_edge("general_chat", "supervisor_node")
workflow.add_edge("manager_agent", "supervisor_node")
workflow.add_edge("judge_agent", "supervisor_node")

workflow.add_edge("responder_agent", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

async def main():
    print(">>> Map-Reduce 并行架构已启动。")
    print(">>> 优化显示模式：屏蔽子任务流式输出，仅显示最终汇总。")
    
    thread_id = "parallel_demo_002"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit"]: break
        
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        # 用于记录哪些节点已经启动过，避免重复打印日志
        started_nodes = set()

        async for event in app.astream_events(input_state, config=config, version="v1"):
            kind = event["event"]
            node_name = event.get("metadata", {}).get("langgraph_node", "")

            # 1. 监控节点启动 (只打印一次)
            if kind == "on_chain_start" and node_name and node_name not in ["__start__", "__end__", "supervisor_node"]:
                if node_name not in started_nodes:
                    print(f"⚡ [{node_name}] 正在后台处理...")
                    started_nodes.add(node_name)

            # 2. 只有 Responder 允许流式输出给用户看 (解决乱码和重复)
            if kind == "on_chat_model_stream":
                if node_name == "responder_agent":
                    chunk = event["data"]["chunk"]
                    if chunk.content: 
                        print(chunk.content, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())