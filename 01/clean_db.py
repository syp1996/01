'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-30 12:38:17
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-30 12:38:44
FilePath: /general_agent/clean_db.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Description: 彻底清理 LangGraph 数据库表 (包括迁移记录)
'''
import asyncio
import os

from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool

load_dotenv()
DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/metro_agent_db")

async def clean():
    print(f"🧹 正在连接数据库: {DB_URI}")
    async with AsyncConnectionPool(conninfo=DB_URI, min_size=1, max_size=1, kwargs={"autocommit": True}) as pool:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                print("💣 正在删除所有 Checkpoint 相关表...")
                # 使用 CASCADE 确保关联表一并删除
                # IF EXISTS 避免报错
                await cur.execute("DROP TABLE IF EXISTS checkpoint_migrations CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS checkpoints CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS checkpoint_blobs CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS checkpoint_writes CASCADE;")
                
                print("✅ 删除完成！数据库已彻底清理。")

if __name__ == "__main__":
    asyncio.run(clean()) 