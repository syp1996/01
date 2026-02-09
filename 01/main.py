'''
Author: Yunpeng Shi
Description: 工业级改造 - 增强型路由逻辑 (深度加固：支持流式多标题解析)
'''
import asyncio
import json
import os
import re
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

# --- 2. 生命周期 ---
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

NODE_DISPLAY_NAMES = {
    "supervisor_node": "总控调度中",
    "ticket_agent": "正在查询票务系统",
    "complaint_agent": "正在处理建议反馈",
    "general_chat": "正在思考",
    "manager_agent": "正在查阅管理手册",
    "judge_agent": "正在查询规章制度",
    "responder_agent": "正在整理回复",
}

# --- 3. 核心 API ---
@app.get("/health")
def health_check():
    return {"status": "ok", "db": "connected"}

@app.get("/threads")
async def list_threads():
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

            is_intermediate = [False] * len(messages)
            for i in range(len(messages)):
                m_type = get_val(messages[i], "type")
                if m_type in ("ai", "assistant"):
                    if i + 1 < len(messages):
                        next_type = get_val(messages[i+1], "type")
                        if next_type in ("ai", "assistant", "tool"):
                            is_intermediate[i] = True
                    m_meta = get_val(messages[i], "metadata", {}) or get_val(messages[i], "response_metadata", {})
                    node = m_meta.get("langgraph_node", "")
                    if node and node != "responder_agent":
                        is_intermediate[i] = True
                    m_name = get_val(messages[i], "name", "")
                    if m_name and m_name != "responder_agent":
                        is_intermediate[i] = True

            for i, msg in enumerate(messages):
                m_type = get_val(msg, "type")
                m_content = get_val(msg, "content", "")
                if m_type in ("human", "user"):
                    if current_ai_msg:
                        history.append(current_ai_msg)
                        current_ai_msg = None
                    history.append({"role": "user", "content": m_content})
                elif m_type in ("ai", "assistant"):
                    if not current_ai_msg:
                        current_ai_msg = {
                            "role": "assistant", "content": "", "thoughts": "",
                            "steps": [], "hasThought": False, "isDoneThinking": True, 
                            "isThoughtExpanded": False 
                        }
                    tool_calls = get_val(msg, "tool_calls", []) or get_val(msg, "additional_kwargs", {}).get("tool_calls", [])
                    if tool_calls:
                        current_ai_msg["hasThought"] = True
                        for tc in tool_calls:
                            name = tc.get("function", {}).get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                            current_ai_msg["steps"].append({"title": f"调用工具: {name}", "status": "done"})
                    if is_intermediate[i]:
                        if m_content:
                            current_ai_msg["hasThought"] = True
                            current_ai_msg["thoughts"] += str(m_content) + "\n"
                    else:
                        if m_content:
                            current_ai_msg["content"] += str(m_content)
                    reasoning = get_val(msg, "additional_kwargs", {}).get("reasoning_content", "")
                    if reasoning:
                        current_ai_msg["hasThought"] = True
                        current_ai_msg["thoughts"] += str(reasoning) + "\n"
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
    try:
        async with app.state.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO thread_metadata (thread_id, title) VALUES (%s, %s) ON CONFLICT (thread_id) DO UPDATE SET title = EXCLUDED.title",
                    (thread_id.strip(), request.title)
                )
                return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
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

# ============================================================================
# ⚠️ 核心流式接口 - 深度加固版 (精准解决 Title/Content 状态切换)
# ============================================================================
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            async with app.state.pool.connection() as conn:
                checkpointer = AsyncPostgresSaver(conn)
                graph_app = build_graph().compile(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": request.thread_id}}
                input_state = {"messages": [HumanMessage(content=request.query)]}
                
                active_steps = set()
                node_state = {} 

                async for event in graph_app.astream_events(input_state, config=config, version="v2"):
                    kind = event["event"]
                    name = event.get("name", "")
                    run_id = event.get("run_id")
                    
                    meta = event.get("metadata", {})
                    node_from_meta = meta.get("langgraph_node", "")
                    tags = event.get("tags", [])
                    is_responder = (node_from_meta == "responder_agent") or ("responder_agent" in tags)

                    if kind == "on_chain_start" and name in NODE_DISPLAY_NAMES:
                        if name != "responder_agent":
                            step_title = NODE_DISPLAY_NAMES.get(name, f"正在运行 {name}")
                            yield format_sse("step", {"title": f"{step_title}...", "status": "loading"})
                            active_steps.add(name)
                            node_state[run_id] = {"buffer": "", "in_content_mode": False}

                    elif kind == "on_chain_end" and name in active_steps:
                        yield format_sse("step", {"title": NODE_DISPLAY_NAMES.get(name, name).replace("正在", "") + " 完成", "status": "done"})
                        active_steps.remove(name)
                        if run_id in node_state: del node_state[run_id]

                    elif kind == "on_tool_start" and name != "FinalAnswer":
                        yield format_sse("step", {"title": f"正在调用工具: {name}...", "status": "loading"})
                    elif kind == "on_tool_end" and name != "FinalAnswer":
                        yield format_sse("step", {"title": f"工具 {name} 调用完成", "status": "done"})
                    
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = chunk.content
                        if not content: continue

                        if is_responder:
                            yield format_sse("message", {"content": content})
                        else:
                            if run_id not in node_state:
                                node_state[run_id] = {"buffer": "", "in_content_mode": False}
                            
                            state = node_state[run_id]
                            state["buffer"] += content
                            
                            while True:
                                buf = state["buffer"]
                                # 1. 尝试寻找新的 Title 块
                                match = re.search(r"Title:\s*(.*?)\s*(?:\n|Content:)(.*)", buf, re.DOTALL)
                                
                                if match:
                                    title = match.group(1).strip()
                                    rest = match.group(2)
                                    # 立即发送步骤更新
                                    yield format_sse("step", {"title": title, "status": "loading"})
                                    state["in_content_mode"] = True
                                    # 处理剩余部分：是否包含下一个 Title:
                                    next_title_match = re.search(r"(.*?)Title:", rest, re.DOTALL)
                                    if next_title_match:
                                        # 当前块内容已全，发送并继续循环
                                        current_content = next_title_match.group(1).strip()
                                        if current_content: yield format_sse("thought", {"content": current_content})
                                        state["buffer"] = rest[len(next_title_match.group(1)):]
                                        state["in_content_mode"] = False
                                        continue
                                    else:
                                        # 进入纯内容模式，发送并清空
                                        if rest.strip(): yield format_sse("thought", {"content": rest.strip()})
                                        state["buffer"] = ""
                                        break
                                
                                # 2. 如果已经处于内容模式，监控是否有新 Title 冒头
                                elif state["in_content_mode"]:
                                    if "Title:" in buf:
                                        idx = buf.find("Title:")
                                        pre = buf[:idx].strip()
                                        if pre: yield format_sse("thought", {"content": pre})
                                        state["buffer"] = buf[idx:]
                                        state["in_content_mode"] = False
                                        continue
                                    else:
                                        # 安全输出当前所有内容
                                        yield format_sse("thought", {"content": buf})
                                        state["buffer"] = ""
                                        break
                                
                                # 3. 降级：缓冲区过大
                                else:
                                    if len(buf) > 300:
                                        yield format_sse("thought", {"content": buf})
                                        state["buffer"] = ""
                                        state["in_content_mode"] = True
                                    break

                # 对话标题生成逻辑...
                final_state = await graph_app.aget_state(config)
                messages = final_state.values.get("messages", [])
                if len(messages) > 0:
                    fq, fa = "", ""
                    for m in messages:
                        if isinstance(m, HumanMessage) and not fq: fq = m.content
                        elif isinstance(m, AIMessage) and not fa and m.content: fa = m.content
                    if fq and fa:
                        from utils import llm
                        try:
                            prompt = f"请根据以下对话提取不超过10个字的简短标题：\n问：{fq[:50]}\n答：{fa[:50]}"
                            gen = await llm.ainvoke([HumanMessage(content=prompt)])
                            title = gen.content.strip().replace('"', '')
                            async with conn.cursor() as cur:
                                await cur.execute("INSERT INTO thread_metadata (thread_id, title) VALUES (%s, %s) ON CONFLICT (thread_id) DO UPDATE SET title = EXCLUDED.title", (request.thread_id, title))
                            yield format_sse("title_generated", {"title": title, "thread_id": request.thread_id})
                        except Exception: pass
                yield format_sse("done", "[DONE]")
        except Exception as e:
            logger.error(f"流式异常: {e}")
            yield format_sse("error", {"error": str(e)})
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)