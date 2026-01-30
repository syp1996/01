from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 导入你的工具
from agents.general_chat import lookup_policy

# ---------------------------------------------------------
# 测试场景 1: RAG 检索工具 (lookup_policy)
# ---------------------------------------------------------

@pytest.mark.asyncio
@patch("utils.get_vector_store")  # ✅ 直接拦截源头模块
async def test_lookup_policy_success(mock_get_store):
    """测试: 当知识库返回正常文档时，工具能否正确格式化输出"""
    mock_store_instance = MagicMock()
    mock_retriever = AsyncMock()
    
    # 构造假文档
    mock_doc_1 = MagicMock()
    mock_doc_1.page_content = "地铁内禁止饮食。"
    mock_doc_1.metadata = {"source_filename": "守则.pdf"}
    
    mock_retriever.ainvoke.return_value = [mock_doc_1]
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store_instance
    
    # 执行
    result = await lookup_policy.ainvoke({"query": "查询规则"})
    
    # 断言
    assert "地铁内禁止饮食" in result
    assert "守则.pdf" in result

@pytest.mark.asyncio
@patch("utils.get_vector_store")  # ✅ 统一 Mock 路径
async def test_lookup_policy_empty(mock_get_store):
    """测试: 当知识库没有相关内容时，是否返回友好提示"""
    mock_store_instance = MagicMock()
    mock_retriever = AsyncMock()
    mock_retriever.ainvoke.return_value = []
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store_instance
    
    result = await lookup_policy.ainvoke({"query": "查询火星"})
    assert result == "未在知识库中找到相关规定。"

@pytest.mark.asyncio
@patch("utils.get_vector_store")  # ✅ 统一 Mock 路径
async def test_lookup_policy_db_error(mock_get_store):
    """测试: 当数据库连接失败时，是否优雅处理"""
    mock_get_store.return_value = None
    result = await lookup_policy.ainvoke({"query": "随便查"})
    assert "系统错误" in result