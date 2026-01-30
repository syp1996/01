from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 这里的引用会自动指向根目录的 agents.general_chat
from agents.general_chat import lookup_policy


@pytest.mark.asyncio
@patch("utils.get_vector_store")  # ✅ 拦截 utils.get_vector_store
async def test_lookup_policy_success(mock_get_store):
    # 1. 构造 Mock 链条
    mock_store_instance = MagicMock()
    mock_retriever = AsyncMock()
    
    # 构造假文档
    mock_doc = MagicMock()
    mock_doc.page_content = "地铁内禁止饮食。"
    mock_doc.metadata = {"source": "manual.pdf"}
    
    # 串联调用关系: get_vector_store() -> store -> as_retriever() -> retriever -> ainvoke() -> [doc]
    mock_get_store.return_value = mock_store_instance
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_retriever.ainvoke.return_value = [mock_doc]
    
    # 2. 执行测试
    result = await lookup_policy.ainvoke({"query": "测试"})
    
    # 3. 断言
    assert "地铁内禁止饮食" in result

@pytest.mark.asyncio
@patch("utils.get_vector_store")
async def test_lookup_policy_empty(mock_get_store):
    # 模拟检索结果为空
    mock_store_instance = MagicMock()
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = [] 
    
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store_instance
    
    result = await lookup_policy.ainvoke({"query": "火星移民"})
    assert "未在知识库中找到相关规定" in result

@pytest.mark.asyncio
@patch("utils.get_vector_store")
async def test_lookup_policy_db_error(mock_get_store):
    # 模拟 get_vector_store 返回 None (初始化失败)
    mock_get_store.return_value = None
    
    result = await lookup_policy.ainvoke({"query": "随便问问"})
    assert "维护中" in result or "系统错误" in result