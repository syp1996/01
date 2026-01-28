import os
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from state import agentState
from utils import complete_current_task, llm

# ==========================================
# 1. 连接 Milvus 向量数据库
# ==========================================

# --- 1.1 环境清理 (与 build_knowledge.py 保持一致) ---
for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "grpc_proxy", "GRPC_PROXY"]:
    if key in os.environ:
        del os.environ[key]
# 确保直连 Docker
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"

# --- 1.2 关键修正：配置参数 ---
# 修正 1: 使用 tcp:// 协议 (刚才验证成功的)
MILVUS_URI = "tcp://127.0.0.1:29530" 
COLLECTION_NAME = "metro_knowledge"
# 修正 2: 指向本地模型路径 (确保离线可用)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"

print(f">>> [General Chat] 正在初始化... (Milvus: {MILVUS_URI}, Model: {LOCAL_MODEL_PATH})")

try:
    # 初始化 Embedding (使用本地模型)
    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_PATH,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 初始化向量库连接
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "uri": MILVUS_URI,
            "token": "",
            "timeout": 30
        },
        auto_id=True
    )
    print(">>> [General Chat] RAG 组件加载成功！")
    
except Exception as e:
    print(f">>> ❌ [General Chat] 初始化失败: {e}")
    # 这里可以抛出异常，或者让后续工具调用时报错
    vector_store = None


# ==========================================
# 2. 定义工具 (Tools)
# ==========================================
@tool
def lookup_policy(query: str) -> str:
    """
    用于查询地铁相关的规章制度、乘客守则、禁止携带物品、票务政策等官方文档。
    当用户的问题涉及具体规定或政策时，必须调用此工具。
    """
    if not vector_store:
        return "系统错误：知识库未正确初始化。"

    try:
        # 检索最相关的 3 个片段
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        if not docs:
            return "未在知识库中找到相关规定。"
        
        # 优化返回格式，增加来源文件名
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source_filename", "未知来源")
            content = doc.page_content.replace('\n', ' ')
            results.append(f"【条款 {i+1}】(来源: {source}): {content}")
            
        return "\n\n".join(results)
        
    except Exception as e:
        return f"系统错误：无法连接知识库服务器 ({str(e)})。请联系管理员。"

# 将工具放入列表
tools = [lookup_policy]


# ==========================================
# 3. 手动构建 ReAct 子图 (Sub-Graph)
# ==========================================
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm_with_tools = llm.bind_tools(tools)

def call_model(state: SubAgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

rag_workflow = StateGraph(SubAgentState)
rag_workflow.add_node("agent", call_model)
rag_workflow.add_node("tools", ToolNode(tools))
rag_workflow.add_edge(START, "agent")
rag_workflow.add_conditional_edges("agent", tools_condition)
rag_workflow.add_edge("tools", "agent")
rag_app = rag_workflow.compile()

# ==========================================
# 4. 主函数
# ==========================================
async def general_chat(state: agentState):
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "general_chat" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
            
    if not isolated_input:
        return {"task_board": board}

    system_prompt = """
    你是一个亲切、专业的地铁综合服务助手。
    你的主要职责是陪乘客闲聊，或者解答一些通用的地铁政策问题。

    策略：
    1. 如果用户只是打招呼或闲聊，直接回复，不要调用工具。
    2. 如果用户问到具体的规定、政策（如携带物品、乘车规则等），**必须调用 lookup_policy 工具**。
    3. 严格根据工具返回的信息回答。如果知识库说没找到，就告诉用户暂不清楚。
    """

    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=isolated_input)
        ]
    }
    
    # 运行子图
    result = await rag_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    updated_board = complete_current_task(state, "general_chat")

    return {
        "messages": [AIMessage(content=final_content, name="general_chat")],
        "task_board": updated_board,
        "task_results": {"general_chat": final_content}
    }