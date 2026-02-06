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
            
            # =========== 【新增】 详细日志打印，用于验证思考过程是否入库 ===========
            print(f"\n[Debug History] Thread ID: {thread_id}, Total Messages: {len(messages)}")
            for i, m in enumerate(messages):
                # 打印消息类型、内容片段和元数据，方便观察是否包含 tool_calls
                content_preview = m.content[:50].replace('\n', ' ') + "..." if m.content else "[No Content]"
                print(f"  [{i}] Type: {m.type:<10} | Content: {content_preview}")
                if hasattr(m, "tool_calls") and m.tool_calls:
                    print(f"       -> Tool Calls: {m.tool_calls}")
            print("=================================================================\n")
            # ===================================================================

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
                
                # 1. 执行对话流
                async for event in graph_app.astream_events(input_state, config=config, version="v2"):
                    kind = event["event"]
                    node_name = event.get("metadata", {}).get("langgraph_node", "")

                    if kind == "on_tool_start":
                        yield format_sse("step", {
                            "title": f"正在调用工具: {event['name']}...",
                            "status": "loading"
                        })
                    elif kind == "on_tool_end":
                        yield format_sse("step", {
                            "title": f"工具 {event['name']} 调用完成",
                            "status": "done"
                        })
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = chunk.content
                        if content:
                            if node_name == "responder_agent":
                                yield format_sse("message", {"content": content})
                            else:
                                yield format_sse("thought", {"content": content})

                # 2. 对话结束：生成智能标题 (修复 Sidebar 随机名问题)
                final_state = await graph_app.aget_state(config)
                messages = final_state.values.get("messages", [])
                
                # 只有在对话轮数较少（通常是第一轮）时才生成标题，避免后续对话覆盖用户自定义的标题
                # 这里的逻辑可以根据需求调整，比如每次都更新，或者只在没有标题时更新
                if len(messages) > 0:
                    # 提取第一轮的问答
                    first_question = ""
                    first_answer = ""
                    
                    for msg in messages:
                        if isinstance(msg, HumanMessage) and not first_question:
                            first_question = msg.content
                        elif isinstance(msg, AIMessage) and not first_answer and msg.content:
                            # 排除掉空的 Tool 调用消息，只取有内容的回答
                            first_answer = msg.content

                    # 调用大模型生成标题 (使用 utils.llm)
                    if first_question and first_answer:
                        from utils import llm  # 延迟导入或确保顶部已导入
                        
                        prompt = f"""
                        请根据以下对话内容，提炼一个极简短的标题（不超过 10 个字）。
                        要求：不要使用标点符号，不要包含"标题"二字，直接返回标题内容。
                        
                        用户：{first_question[:200]}
                        回答：{first_answer[:200]}
                        """
                        try:
                            # 使用非流式调用生成标题
                            generated_title_msg = await llm.ainvoke([HumanMessage(content=prompt)])
                            title = generated_title_msg.content.strip().replace('"', '').replace('“', '').replace('”', '')
                            
                            # 3. 持久化标题到数据库 (thread_metadata 表)
                            # 注意：conn 是外层 context manager 的，这里可以直接用 cursor
                            async with conn.cursor() as cur:
                                await cur.execute(
                                    """
                                    INSERT INTO thread_metadata (thread_id, title) VALUES (%s, %s)
                                    ON CONFLICT (thread_id) DO UPDATE SET title = EXCLUDED.title
                                    """,
                                    (request.thread_id, title)
                                )
                                logger.info(f"标题已更新: {title} (ID: {request.thread_id})")
                            
                            # 4. 推送标题更新事件给前端 (让前端自动刷新 Sidebar Item)
                            yield format_sse("title_generated", {"title": title, "thread_id": request.thread_id})
                            
                        except Exception as e:
                            logger.error(f"生成标题失败: {e}")

                # 5. 发送结束信号
                yield format_sse("done", "[DONE]")

        except Exception as e:
            logger.error(f"流式异常: {e}")
            yield format_sse("error", {"error": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)