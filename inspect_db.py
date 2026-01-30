import asyncio
import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# 加载环境变量
load_dotenv()
DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/metro_agent_db")

async def inspect():
    print(f"🔍 正在连接数据库: {DB_URI}")
    async with AsyncConnectionPool(conninfo=DB_URI, max_size=1, kwargs={"autocommit": True}) as pool:
        async with pool.connection() as conn:
            # 使用与 main.py 相同的配置 (默认配置)
            checkpointer = AsyncPostgresSaver(conn)
            
            # 1. 列出所有会话
            print("\n📋 最近活跃的会话 (Threads):")
            # 直接查询底层表（虽然是二进制，但 thread_id 是文本）
            async with conn.cursor() as cur:
                await cur.execute("SELECT DISTINCT thread_id FROM checkpoints LIMIT 10")
                threads = await cur.fetchall()
                if not threads:
                    print("   (暂无数据)")
                    return
                for t in threads:
                    print(f"   - {t[0]}")
                
                target_thread = threads[0][0] # 取第一个线程来分析

            # 2. 读取该会话的最新状态
            print(f"\n🕵️‍♂️ 正在分析会话 [{target_thread}] 的最新记忆...")
            # 使用 LangGraph 提供的 api 来读取，它会自动帮我们反序列化 Msgpack
            config = {"configurable": {"thread_id": target_thread}}
            checkpoint = await checkpointer.aget(config)
            
            if not checkpoint:
                print("   ❌ 未找到 Checkpoint 数据")
            else:
                print("   ✅ 数据读取成功！")
                # 提取 messages
                channel_values = checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])
                
                print(f"   📊 包含消息数: {len(messages)}")
                for i, msg in enumerate(messages):
                    # 打印消息类型和内容
                    msg_type = msg.__class__.__name__
                    content = getattr(msg, "content", "")[:50] + "..." # 只显示前50字
                    print(f"      [{i}] {msg_type}: {content}")

                # 提取任务看板
                board = channel_values.get("task_board", [])
                if board:
                    print(f"\n   📋 任务看板 ({len(board)} 个任务):")
                    for task in board:
                        print(f"      - [{task.get('status')}] {task.get('description')}")

if __name__ == "__main__":
    asyncio.run(inspect())