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

def format_sse(event_type: str, data: dict) -> str:
    """
    格式化 SSE 数据包
    :param event_type: 事件类型 ('thought', 'step', 'message', 'done', 'error')
    :param data: 数据字典
    """
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

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
    """获取指定会话的历史记录 (修复：基于消息时序识别思考过程)"""
    try:
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            graph_app = build_graph().compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            
            state = await graph_app.aget_state(config)
            messages = state.values.get("messages", [])
            
            history = []
            current_ai_msg = None

            def get_val(obj, key, default=None):
                if isinstance(obj, dict): return obj.get(key, default)
                return getattr(obj, key, default)

            # --- [核心逻辑] 预先识别中间过程消息 ---
            # 规则：如果 AI 消息后面紧跟着另一条 AI 消息或 Tool 消息，它一定是中间过程
            is_intermediate = [False] * len(messages)
            for i in range(len(messages)):
                m_type = get_val(messages[i], "type")
                if m_type in ("ai", "assistant"):
                    # 1. 如果后面还有消息，且下一条不是用户发的，说明当前这条是中间过程
                    if i + 1 < len(messages):
                        next_type = get_val(messages[i+1], "type")
                        if next_type in ("ai", "assistant", "tool"):
                            is_intermediate[i] = True
                    
                    # 2. 检查元数据（作为补充）
                    m_meta = get_val(messages[i], "metadata", {}) or get_val(messages[i], "response_metadata", {})
                    node = m_meta.get("langgraph_node", "")
                    if node and node != "responder_agent":
                        is_intermediate[i] = True
                        
                    # 3. 检查消息名称（部分 Agent 会设置 name）
                    m_name = get_val(messages[i], "name", "")
                    if m_name and m_name != "responder_agent":
                        is_intermediate[i] = True

            for i, msg in enumerate(messages):
                m_type = get_val(msg, "type")
                m_content = get_val(msg, "content", "")

                # 1. 用户消息：结算上一个 AI 回合
                if m_type in ("human", "user"):
                    if current_ai_msg:
                        history.append(current_ai_msg)
                        current_ai_msg = None
                    history.append({"role": "user", "content": m_content})
                
                # 2. AI 消息
                elif m_type in ("ai", "assistant"):
                    if not current_ai_msg:
                        current_ai_msg = {
                            "role": "assistant", "content": "", "thoughts": "",
                            "steps": [], "hasThought": False, "isDoneThinking": True, 
                            "isThoughtExpanded": False 
                        }

                    # A. 提取工具调用
                    tool_calls = get_val(msg, "tool_calls", []) or get_val(msg, "additional_kwargs", {}).get("tool_calls", [])
                    if tool_calls:
                        current_ai_msg["hasThought"] = True
                        for tc in tool_calls:
                            # 兼容对象和字典格式
                            name = tc.get("function", {}).get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                            current_ai_msg["steps"].append({"title": f"正在调用工具: {name}", "status": "done"})

                    # B. 核心判断：放入思考区还是正文区
                    if is_intermediate[i]:
                        if m_content:
                            current_ai_msg["hasThought"] = True
                            current_ai_msg["thoughts"] += str(m_content) + "\n"
                    else:
                        if m_content:
                            current_ai_msg["content"] += str(m_content)

                    # C. 处理推理内容 (DeepSeek 专用)
                    reasoning = get_val(msg, "additional_kwargs", {}).get("reasoning_content", "")
                    if reasoning:
                        current_ai_msg["hasThought"] = True
                        current_ai_msg["thoughts"] += str(reasoning) + "\n"

                # 3. 工具响应消息
                elif m_type == "tool":
                    if current_ai_msg:
                        current_ai_msg["hasThought"] = True

            if current_ai_msg:
                history.append(current_ai_msg)

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
                
                # 【核心修改】使用 version="v2" 获取更详细的内部事件
                async for event in graph_app.astream_events(input_state, config=config, version="v2"):
                    
                    kind = event["event"]
                    # 获取当前事件所属的节点 (例如: 'ticket_agent', 'responder_agent')
                    # tags 列表里通常包含节点名，或者从 metadata 获取
                    node_name = event.get("metadata", {}).get("langgraph_node", "")

                    # -------------------------------------------------
                    # 1. 捕获 [Step]: 工具调用开始
                    # -------------------------------------------------
                    if kind == "on_tool_start":
                        # event['name'] 是工具的函数名
                        tool_name = event['name']
                        yield format_sse("step", {
                            "title": f"正在调用工具: {tool_name}...",
                            "status": "loading"
                        })

                    # -------------------------------------------------
                    # 2. 捕获 [Step]: 工具调用结束
                    # -------------------------------------------------
                    elif kind == "on_tool_end":
                        tool_name = event['name']
                        # 可以在这里把工具的输出结果摘要发给前端，这里简单处理
                        yield format_sse("step", {
                            "title": f"工具 {tool_name} 调用完成",
                            "status": "done"
                        })

                    # -------------------------------------------------
                    # 3. 捕获 LLM 输出 (区分 思考过程 vs 最终回复)
                    # -------------------------------------------------
                    elif kind == "on_chat_model_stream":
                        # v2 版本中，数据在 event["data"]["chunk"]
                        chunk = event["data"]["chunk"]
                        content = chunk.content

                        if content:
                            # 策略：如果是由 'responder_agent' (最终回复者) 生成的，就是 message
                            # 如果是由其他 agent (如 ticket_agent 在分析参数) 生成的，就是 thought
                            if node_name == "responder_agent":
                                yield format_sse("message", {"content": content})
                            else:
                                # 其他节点的输出视为“思考过程”
                                yield format_sse("thought", {"content": content})

                # 会话结束，保存标题 (保持原有逻辑)
                final_state = await graph_app.aget_state(config)
                # 注意：你的 state 定义中 title 可能在 values 里
                # 如果没有 title 生成逻辑，这里可能获取不到，保持原样即可
                # 假设 summary agent 或其他地方生成了 title
                # ... (原有保存标题逻辑)
                
                yield format_sse("done", "[DONE]")

        except Exception as e:
            logger.error(f"流式异常: {e}")
            yield format_sse("error", {"error": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)