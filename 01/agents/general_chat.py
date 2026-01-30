'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/general_chat.py
Description: 并行化改造版 - 修复 TypedDict 调用表达式报错 (最终稳定版)
'''
import os
from typing import Annotated, List, TypedDict

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# agents/general_chat.py 顶部
from utils import complete_current_task  # ⚠️ 确保加上 get_vector_store
from utils import get_vector_store, llm, update_task_result

from state import WorkerState

# --- 全局变量先设为 None ---
vector_store = None
embeddings = None

def get_vector_store():
    """懒加载函数：只有在真正需要的时候才初始化连接"""
    global vector_store, embeddings
    
    # 如果已经初始化过，直接返回
    if vector_store is not None:
        return vector_store

    print(f">>> [General Chat] 正在初始化 Milvus 连接... (延迟加载)")
    
    # 这里定义配置
    MILVUS_URI = "tcp://127.0.0.1:29530" 
    COLLECTION_NAME = "metro_knowledge"
    LOCAL_MODEL_PATH = "./models/bge-small-zh-v1.5"

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": MILVUS_URI,
                "token": "",
                "timeout": 30
            },
            index_params={"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}},
            auto_id=True
        )
        print(">>> [General Chat] RAG 组件加载成功！")
        return vector_store
        
    except Exception as e:
        print(f">>> ❌ [General Chat] 初始化失败: {e}")
        return None

@tool
async def lookup_policy(query: str) -> str:
    """查询地铁相关规章制度、乘车守则等官方文档。"""
    
    # 在主事件循环中获取 store，这下有 Loop 了
    store = get_vector_store()
    
    if not store:
        return "系统错误：知识库未正确初始化（请检查 Milvus 服务或控制台报错日志）。"

    try:
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5, 
                "score_threshold": 0.4,
                "param": {"metric_type": "L2", "nprobe": 10} 
            }
        )
        
        # 2. 检索也建议改为异步调用 (await .ainvoke)
        docs = await retriever.ainvoke(query)
        
        if not docs:
            return "未在知识库中找到相关规定。"
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source_filename', '未知')
            clean_content = doc.page_content.replace('\n', ' ')
            results.append(f"【条款 {i+1}】(来源: {source}): {clean_content}")
            
        return "\n\n".join(results)
    except Exception as e:
        return f"系统错误：知识库检索失败 ({str(e)})。"
# --- 2. ReAct 子图定义 (修复报错的关键部分) ---

tools = [lookup_policy]
llm_with_tools = llm.bind_tools(tools)

# 【修复】显式定义 TypedDict 类，而不是在函数参数里调用构造函数
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def call_model(state: SubAgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 【修复】使用定义好的类
rag_workflow = StateGraph(SubAgentState)
rag_workflow.add_node("agent", call_model)
rag_workflow.add_node("tools", ToolNode(tools))
rag_workflow.add_edge(START, "agent")
rag_workflow.add_conditional_edges("agent", tools_condition)
rag_workflow.add_edge("tools", "agent")
rag_app = rag_workflow.compile()

# --- 3. 主函数 ---

async def general_chat(state: WorkerState):
    task = state["task"]
    isolated_input = task['input_content']

    # 【核心逻辑】获取并处理历史
    global_messages = state.get("messages", [])
    print(f"[General] 正在处理 (RAG已启用): {isolated_input}")

    # 技巧：全局历史的最后一条通常是用户本轮的“复杂指令”（被 Supervisor 拆解前的）。
    # 为了让 Worker 专注处理 isolated_input，我们通常取“上一轮为止的历史”作为 Context。
    history_context = global_messages[:-1] if global_messages else []

    # 强化 Prompt (Few-Shot)
    system_prompt = """你是一个亲切、专业的地铁综合服务助手。
    你的主要职责是陪乘客闲聊，或者依据真实规定解答地铁政策问题。

    ### 核心指令（必须严格遵守）：
    1. **必须查证**：当用户问到具体的规定、政策（如携带物品、安检、票务规则）时，**必须调用 lookup_policy 工具**。
    2. **强制标记**：凡是你的回答中引用了 `lookup_policy` 工具返回的信息，**必须**在对应的句子末尾加上 `【📚知识库】` 标记。这是为了向用户证明信息的权威性。
    
    ### 示例学习（请模仿）：
    - 用户：能带白酒进站吗？
    - 工具返回：...50度以上散装白酒禁止携带...
    - ❌ 错误回答：根据规定，散装的高浓度白酒是不让带的。
    - ✅ 正确回答：为您查询了相关规定，50度以上的散装白酒是禁止携带进站的【📚知识库】。

    3. **诚实原则**：如果工具未找到相关信息，请直接告诉用户“暂未查到相关规定”，这种情况下**不需要**加标记。
    """

    # 【构造输入】 System + 历史Context + 当前纯净指令
    inputs = {
        "messages": [SystemMessage(content=system_prompt)] + history_context + [HumanMessage(content=isolated_input)]
    }
    result = await rag_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    return {
        "messages": [AIMessage(content=final_content, name="general_chat")],
        "task_board": [update_task_result(task, result=final_content)]
    }