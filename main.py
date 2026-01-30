'''
Author: Yunpeng Shi
Description: 工业级改造 - FastAPI 服务端 + Postgres 持久化 (稳定版 - 使用默认二进制存储)
'''
import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# 导入你的智能体
from agents.complaint_agent import complaint_agent
from agents.general_chat import general_chat
from agents.judge_agent import judge_agent
from agents.manager_agent import manager_agent
from agents.responder_agent import responder_agent
from agents.supervisor import supervisor_node, workflow_router
from agents.ticket_agent import ticket_agent
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
# --- LangGraph & Postgres 核心组件 ---
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from state import agentState

load_dotenv()

# --- 配置 ---
DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/metro_agent_db")

# --- 1. 构建图 (保持不变) ---
def build_graph():
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
            "ticket_agent", "complaint_agent", "general_chat", 
            "manager_agent", "judge_agent", "responder_agent"
        ]
    )
    workflow.add_edge("ticket_agent", "supervisor_node")
    workflow.add_edge("complaint_agent", "supervisor_node")
    workflow.add_edge("general_chat", "supervisor_node")
    workflow.add_edge("manager_agent", "supervisor_node")
    workflow.add_edge("judge_agent", "supervisor_node")
    workflow.add_edge("responder_agent", END)
    
    return workflow

# --- 2. 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> 正在初始化数据库连接池...")
    
    # kwargs={"autocommit": True} 是必须的，确保事务自动提交
    async with AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True}) as pool:
        app.state.pool = pool
        
        print(">>> 正在检查 Checkpoint 表结构 (使用默认二进制存储)...")
        async with pool.connection() as conn:
            # 【关键修改】不传 serde 参数 -> 使用默认的 Msgpack (二进制)
            # 这会自动创建或检查 checkpoints 表（此时 checkpoint 列为 bytea 类型）
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
        
        print(">>> 数据库连接成功，服务已就绪。")
        yield
        
    print(">>> 正在关闭数据库连接...")

# --- 3. 初始化 FastAPI ---
app = FastAPI(title="Metro AI Agent Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"

# --- 4. 核心 API 接口 ---
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式接口
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        async with app.state.pool.connection() as conn:
            # 【关键修改】保持一致，不传 serde 参数
            checkpointer = AsyncPostgresSaver(conn)
            
            workflow = build_graph()
            graph_app = workflow.compile(checkpointer=checkpointer)
            
            config = {"configurable": {"thread_id": request.thread_id}}
            input_state = {"messages": [HumanMessage(content=request.query)]}
            
            started_nodes = set()

            try:
                # 使用 v1 版本事件流
                async for event in graph_app.astream_events(input_state, config=config, version="v1"):
                    kind = event["event"]
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    
                    # 1. 节点启动事件
                    if kind == "on_chain_start" and node_name and node_name not in ["__start__", "__end__", "supervisor_node"]:
                        if node_name not in started_nodes:
                            started_nodes.add(node_name)
                            yield f"event: agent_start\ndata: {json.dumps({'agent': node_name})}\n\n"

                    # 2. 消息流式输出事件
                    if kind == "on_chat_model_stream":
                        if node_name == "responder_agent":
                            chunk = event["data"]["chunk"]
                            if chunk.content:
                                # ensure_ascii=False 解决中文乱码
                                yield f"event: message\ndata: {json.dumps({'content': chunk.content}, ensure_ascii=False)}\n\n"
                
                yield "event: done\ndata: [DONE]\n\n"
                
            except Exception as e:
                print(f"Server Error: {str(e)}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    return {"status": "ok", "db": "connected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)