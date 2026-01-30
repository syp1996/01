from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 导入你的工具
from agents.general_chat import lookup_policy

# ---------------------------------------------------------
# 测试场景 1: RAG 检索工具 (lookup_policy)
# ---------------------------------------------------------

# ⚠️ 修正点：统一 Patch 路径为 'agents.general_chat.get_vector_store'
# 因为 general_chat.py 既然 import 了它，我们就要拦截那个 import 进来的副本
@pytest.mark.asyncio
@patch("agents.general_chat.get_vector_store")
async def test_lookup_policy_success(mock_get_store):
    """
    测试: 当知识库返回正常文档时，工具能否正确格式化输出？
    """
    # 1. 构造 Mock 数据库实例
    mock_store_instance = MagicMock()
    mock_retriever = AsyncMock()
    
    # 2. 构造假文档 (模拟 RAG 检索结果)
    mock_doc_1 = MagicMock()
    mock_doc_1.page_content = "地铁内禁止饮食。"
    mock_doc_1.metadata = {"source_filename": "守则.pdf"}
    
    mock_doc_2 = MagicMock()
    mock_doc_2.page_content = "携带折叠自行车需折叠。"
    mock_doc_2.metadata = {"source_filename": "规定.txt"}
    
    # 3. 设置 Mock 返回值
    mock_retriever.ainvoke.return_value = [mock_doc_1, mock_doc_2]
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store_instance
    
    # 4. 执行被测工具
    # 使用 .ainvoke 调用，并传入字典参数 (LangChain Tool 的标准调用方式)
    result = await lookup_policy.ainvoke({"query": "查询规则"})
    
    # 5. 断言检查
    # 检查返回的字符串中是否包含我们 Mock 的内容
    assert "地铁内禁止饮食" in result
    assert "守则.pdf" in result
    assert "【条款 1】" in result

@pytest.mark.asyncio
@patch("agents.general_chat.get_vector_store")
async def test_lookup_policy_empty(mock_get_store):
    """
    测试: 当知识库没有相关内容时，是否返回了友好的提示？
    """
    mock_store_instance = MagicMock()
    mock_retriever = AsyncMock()
    
    # 模拟检索结果为空
    mock_retriever.ainvoke.return_value = []
    
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store_instance
    
    # 执行
    result = await lookup_policy.ainvoke({"query": "查询火星移民计划"})
    
    # 断言
    assert result == "未在知识库中找到相关规定。"

@pytest.mark.asyncio
@patch("agents.general_chat.get_vector_store")
async def test_lookup_policy_db_error(mock_get_store):
    """
    测试: 当数据库连接失败（返回 None）时，是否优雅处理？
    """
    # 模拟 DB 初始化失败 (get_vector_store 返回 None)
    mock_get_store.return_value = None
    
    # 执行
    result = await lookup_policy.ainvoke({"query": "随便查查"})
    
    # 断言
    assert "系统错误" in result