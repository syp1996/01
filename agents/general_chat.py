import os
from typing import Annotated, List, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
# --- LangGraph 原生构建模块 ---
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from state import agentState
from utils import complete_current_task, llm

# ==========================================
# 1. 初始化资源 (向量库加载)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(CURRENT_DIR, "..", "data", "vector_store")
INDEX_NAME = "metro_knowledge"

# 必须与构建时使用的模型一致
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

try:
    if os.path.exists(VECTOR_DB_DIR):
        print(f">>> [General Chat] 正在加载本地知识库: {VECTOR_DB_DIR}")
        vector_store = FAISS.load_local(
            VECTOR_DB_DIR, 
            embeddings, 
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        print(">>> [General Chat] 本地知识库加载成功。")
    else:
        print(f">>> [General Chat] 警告：未找到向量库目录 {VECTOR_DB_DIR}")
except Exception as e:
    print(f">>> [General Chat] 知识库加载失败: {e}")


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
        return "知识库系统维护中，暂时无法查询详细规定。"
    
    # 检索最相关的 3 个片段
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    if not docs:
        return "未在知识库中找到相关规定。"
    
    results = [f"【相关条款 {i+1}】：{doc.page_content}" for i, doc in enumerate(docs)]
    return "\n\n".join(results)

# 将工具放入列表
tools = [lookup_policy]


# ==========================================
# 3. 手动构建 ReAct 子图 (Sub-Graph)
# ==========================================

# 3.1 定义子图的状态（仅包含消息列表即可）
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 3.2 绑定工具到 LLM
llm_with_tools = llm.bind_tools(tools)

# 3.3 定义节点函数：调用模型
def call_model(state: SubAgentState):
    # 调用模型，模型会根据 history 决定是说话还是调工具
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 3.4 构建图结构
rag_workflow = StateGraph(SubAgentState)

# 添加节点
rag_workflow.add_node("agent", call_model)
rag_workflow.add_node("tools", ToolNode(tools)) # LangGraph 内置的工具执行节点

# 设置连线
rag_workflow.add_edge(START, "agent")

# 【关键】条件边：agent 跑完后，看结果是 ToolCall 还是普通文本
# 如果是 ToolCall -> 去 "tools" 节点
# 如果是 文本 -> 去 END
rag_workflow.add_conditional_edges(
    "agent",
    tools_condition
)

# 工具跑完后，结果必须回传给 agent 继续思考
rag_workflow.add_edge("tools", "agent")

# 编译子图
rag_app = rag_workflow.compile()


# ==========================================
# 4. 主函数 (Worker Node)
# ==========================================

async def general_chat(state: agentState):
    # 1. 提取任务输入
    board = state.get("task_board", [])
    isolated_input = ""
    for task in board:
        if task['task_type'] == "general_chat" and task['status'] == 'pending':
            isolated_input = task['input_content']
            break
            
    if not isolated_input:
        return {"task_board": board}

    # 2. 准备 Prompt
    system_prompt = """
    你是一个亲切、专业的地铁综合服务助手。
    你的主要职责是陪乘客闲聊，或者解答一些通用的地铁政策问题。

    策略：
    1. 如果用户只是打招呼或闲聊（如“你好”），请直接热情回复，**不要调用工具**。
    2. 如果用户问到具体的规定（如“能带酒吗”、“能带狗吗”），**必须调用 lookup_policy 工具**搜索知识库。
    3. 根据工具返回的信息回答问题。如果工具说没查到，就诚实地告诉用户你暂时不清楚。
    """

    # 3. 构造子图输入
    # 我们把 SystemPrompt 和 用户输入 封装成初始消息列表
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=isolated_input)
        ]
    }
    
    # 4. 运行 ReAct 子图
    result = await rag_app.ainvoke(inputs)
    
    # 5. 获取最终回复 (最后一条消息的内容)
    final_content = result["messages"][-1].content

    # 6. 销账
    updated_board = complete_current_task(state, "general_chat")

    return {
        "messages": [AIMessage(content=final_content, name="general_chat")],
        "task_board": updated_board,
        "task_results": {"general_chat": final_content}
    }