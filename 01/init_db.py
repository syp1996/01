'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-30 11:46:38
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-30 11:48:20
FilePath: /general_agent/01/init_db.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Description: 手动初始化数据库表结构 (修正连接池参数版)
'''
import asyncio
import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# 加载环境变量
load_dotenv()
DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/metro_agent_db")

async def init():
    print(f"🔌 正在连接数据库: {DB_URI}")
    try:
        # 【关键修改】显式设置 min_size=1，配合 max_size=1
        async with AsyncConnectionPool(
            conninfo=DB_URI, 
            min_size=1,       # <--- 新增这行
            max_size=1, 
            kwargs={"autocommit": True}
        ) as pool:
            print(">>> 连接池已建立")
            async with pool.connection() as conn:
                print("🛠️  正在执行 checkpointer.setup() 建表...")
                
                # 使用默认配置 (Msgpack 二进制存储)
                checkpointer = AsyncPostgresSaver(conn)
                await checkpointer.setup()
                
                print("✅ 建表成功！checkpoints 表已就绪。")
                
                # 验证
                async with conn.cursor() as cur:
                    await cur.execute("SELECT count(*) FROM checkpoints")
                    count = await cur.fetchone()
                    print(f"📊 当前表验证通过，记录数: {count[0]}")
                    
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == "__main__":
    asyncio.run(init())