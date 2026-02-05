'''
Author: Yunpeng Shi
Description: 工业级改造 - FastAPI 服务端 + Postgres 持久化 (支持标题、历史记录、删除)
'''
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

from agents.complaint_agent import complaint_agent
from agents.general_chat import general_chat
from agents.judge_agent import judge_agent
from agents.manager_agent import manager_agent
from agents.responder_agent import responder_agent
from agents.supervisor import supervisor_node, workflow_router
from agents.ticket_agent import ticket_agent
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from state import agentState
from utils import logger

load_dotenv()
DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/metro_agent_db")

# --- 1. 构建智能体图 ---
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
        ["ticket_agent", "complaint_agent", "general_chat", "manager_agent", "judge_agent", "responder_agent"]
    )
    workflow.add_edge("ticket_agent", "supervisor_node")
    workflow.add_edge("complaint_agent", "supervisor_node")
    workflow.add_edge("general_chat", "supervisor_node")
    workflow.add_edge("manager_agent", "supervisor_node")
    workflow.add_edge("judge_agent", "supervisor_node")
    workflow.add_edge("responder_agent", END)
    return workflow

# --- 2. 生命周期与数据库池 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>> 正在初始化数据库连接池...")
    async with AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True}) as pool:
        app.state.pool = pool
        async with pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
        logger.info(">>> 服务启动成功，路由已就绪。")
        yield
    logger.info(">>> 服务已停止。")

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

class RenameRequest(BaseModel):
    title: str

# --- 3. 核心 API 接口 ---

@app.get("/health")
def health_check():
    return {"status": "ok", "db": "connected"}

@app.get("/threads")
async def list_threads():
    """获取会话列表 (关联标题)"""
    try:
        async with app.state.pool.connection() as conn:
            async with conn.cursor() as cur:
                query = """
                SELECT c.thread_id, COALESCE(m.title, c.thread_id) as title
                FROM (SELECT DISTINCT thread_id FROM checkpoints) c
                LEFT JOIN thread_metadata m ON c.thread_id = m.thread_id
                ORDER BY c.thread_id DESC
                """
                await cur.execute(query)
                rows = await cur.fetchall()
                return [{"thread_id": row[0], "title": row[1]} for row in rows]
    except Exception as e:
        logger.error(f"获取列表失败: {e}")
        return []

@app.get("/threads/{thread_id}/history")
async def get_history(thread_id: str):
    """获取指定会话的历史记录"""
    try:
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            graph_app = build_graph().compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            
            state = await graph_app.aget_state(config)
            messages = state.values.get("messages", [])
            
            history = []
            for msg in messages:
                # 统一转换角色名为前端可读的 user/assistant
                if msg.type in ("human", "user"):
                    history.append({"role": "user", "content": msg.content})
                elif msg.type in ("ai", "assistant"):
                    history.append({"role": "assistant", "content": msg.content})
            return {"history": history}
    except Exception as e:
        logger.error(f"获取历史失败: {e}")
        return {"history": []}

@app.post("/threads/{thread_id}/rename")
async def rename_thread(thread_id: str, request: RenameRequest):
    """重命名会话标题"""
    try:
        async with app.state.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO thread_metadata (thread_id, title) VALUES (%s, %s)
                    ON CONFLICT (thread_id) DO UPDATE SET title = EXCLUDED.title
                    """,
                    (thread_id.strip(), request.title)
                )
                return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """删除会话"""
    try:
        async with app.state.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM thread_metadata WHERE thread_id = %s", (thread_id,))
                await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            async with app.state.pool.connection() as conn:
                checkpointer = AsyncPostgresSaver(conn)
                graph_app = build_graph().compile(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": request.thread_id}}
                input_state = {"messages": [HumanMessage(content=request.query)]}
                
                async for event in graph_app.astream_events(input_state, config=config, version="v1"):
                    if event["event"] == "on_chat_model_stream" and event.get("metadata", {}).get("langgraph_node") == "responder_agent":
                        chunk = event["data"]["chunk"]
                        if chunk.content:
                            yield f"event: message\ndata: {json.dumps({'content': chunk.content}, ensure_ascii=False)}\n\n"
                
                # 会话结束，保存标题
                final_state = await graph_app.aget_state(config)
                title = final_state.values.get("title")
                if title:
                    async with app.state.pool.connection() as conn:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                "INSERT INTO thread_metadata (thread_id, title) VALUES (%s, %s) ON CONFLICT (thread_id) DO NOTHING",
                                (request.thread_id, title)
                            )
                yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式异常: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)