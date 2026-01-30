import os

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    """
    在所有测试执行前，自动设置虚拟的环境变量。
    防止测试代码连接真实的 DeepSeek 或 Milvus。
    """
    # 保存旧的环境变量（如果有）
    old_environ = dict(os.environ)
    
    # 设置测试用的环境变量
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-dummy-key"
    os.environ["DEEPSEEK_BASE_URL"] = "https://api.test.com"
    os.environ["DB_URI"] = "postgresql://test:test@localhost:5432/test_db"
    
    yield
    
    # 测试结束后恢复现场
    os.environ.clear()
    os.environ.update(old_environ)