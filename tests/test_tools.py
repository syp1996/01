from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.general_chat import lookup_policy


@pytest.mark.asyncio
@patch("utils.get_vector_store")  # ✅ 统一 patch 模块源头
async def test_lookup_policy_success(mock_get_store):
    # 1. 构造 Mock 替身
    mock_store_instance = MagicMock()
    mock_retriever = AsyncMock()
    
    mock_doc = MagicMock()
    mock_doc.page_content = "地铁内禁止饮食。"
    mock_doc.metadata = {"source": "manual.pdf"}
    
    mock_retriever.ainvoke.return_value = [mock_doc]
    mock_store_instance.as_retriever.return_value = mock_retriever
    mock_get_store.return_value = mock_store_instance
    
    # 2. 执行
    result = await lookup_policy.ainvoke({"query": "测试"})
    
    # 3. 断言 (现在一定会拿到假数据了)
    assert "地铁内禁止饮食" in result

# ... (以此类推修改 test_lookup_policy_empty 和 test_lookup_policy_db_error，全部 patch "utils.get_vector_store")    assert "地铁内禁止饮食" in result

# ... (以此类推修改 test_lookup_policy_empty 和 test_lookup_policy_db_error，全部 patch "utils.get_vector_store")