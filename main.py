'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-26 08:49:23
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-27 11:07:16
FilePath: /01/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.complaint_agent import complaint_agent
from agents.general_chat import general_chat
from agents.judge_agent import judge_agent
from agents.manager_agent import manager_agent
from agents.responder_agent import responder_agent
# 2. 导入所有智能体
from agents.supervisor import supervisor_node
from agents.ticket_agent import ticket_agent
# 1. 导入基础模块
from state import agentState
from utils import WORKERS_INFO

load_dotenv()

# --- 边逻辑函数 ---
def get_next_node(state: agentState) -> str:
    return state.get("next_step", "FINISH")

# --- 构建图 ---
workflow = StateGraph(agentState)

# 1. 添加节点
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("ticket_agent", ticket_agent)
workflow.add_node("complaint_agent", complaint_agent)
workflow.add_node("general_chat", general_chat)
workflow.add_node("manager_agent", manager_agent)
workflow.add_node("judge_agent", judge_agent)
workflow.add_node("responder_agent", responder_agent)

# 2. 设置连线
workflow.add_edge(START, 'supervisor_node')

# 3. 设置条件边
# 动态构建映射表，更加优雅
conditional_map = {k: k for k in WORKERS_INFO.keys()}
conditional_map["FINISH"] = "responder_agent"

workflow.add_conditional_edges(
    "supervisor_node",
    get_next_node,
    conditional_map
)

# 4. 设置回环（Worker -> Supervisor）
for key in WORKERS_INFO.keys():
    workflow.add_edge(key, 'supervisor_node')

workflow.add_edge("responder_agent", END)

# --- 编译与运行 ---
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

async def main():
    print(">>> 模块化微服务架构已启动。")
    thread_id = "modular_test_001"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit"]: break
        
        input_state = {"messages": [HumanMessage(content=user_input)]}

        async for event in app.astream_events(input_state, config=config, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name == "responder_agent":
                    chunk = event["data"]["chunk"]
                    if chunk.content: print(chunk.content, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())