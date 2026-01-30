'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/general_chat.py
Description: 并行化改造版 - 修复 TypedDict 调用表达式报错 (最终稳定版)
'''
import os
from typing import Annotated, List, TypedDict

# 1. 导入 utils 模块
import utils
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.tools import tool
# ⚠️ 已删除：本地不再需要引入 Milvus 和 HF，逻辑全部收敛到 utils
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# 2. 显式导入需要使用的函数，确保测试能 Patch 到它们
from utils import (complete_current_task, get_vector_store, llm,
                   update_task_result)

from state import WorkerState

# ⚠️ 已删除：本地定义的 get_vector_store 及其全局变量
# 现在的逻辑是：直接使用从 utils 导入的 get_vector_store
# 这样测试脚本里的 @patch("utils.get_vector_store") 才能生效

@tool
async def lookup_policy(query: str) -> str:
    """查询地铁相关规章制度、乘车守则等官方文档。"""
    
    # 3. 这里调用的是 utils.get_vector_store() (虽然写法上没带前缀，但因为它被 from utils import... 导入了)
    # 测试环境会拦截这个调用，返回 Mock 对象；生产环境会调用 utils 里的真实逻辑。
    store = get_vector_store()
    
    if not store:
        return "系统错误：知识库未正确初始化（请检查 Milvus 服务或控制台报错日志）。"

    try:
        # 获取检索器
        retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5, 
                "score_threshold": 0.4,
                "param": {"metric_type": "L2", "nprobe": 10} 
            }
        )
        
        # 异步调用检索
        docs = await retriever.ainvoke(query)
        
        if not docs:
            return "未在知识库中找到相关规定。"
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source_filename', '未知')
            # 清洗换行符，防止输出格式混乱
            clean_content = doc.page_content.replace('\n', ' ')
            results.append(f"【条款 {i+1}】(来源: {source}): {clean_content}")
            
        return "\n\n".join(results)
    except Exception as e:
        return f"系统错误：知识库检索失败 ({str(e)})。"

# --- 2. ReAct 子图定义 ---

tools = [lookup_policy]
llm_with_tools = llm.bind_tools(tools)

# 定义 State 类型
class SubAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def call_model(state: SubAgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 构建图
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

    # 获取并处理历史
    global_messages = state.get("messages", [])
    print(f"[General] 正在处理 (RAG已启用): {isolated_input}")

    # 取“上一轮为止的历史”作为 Context
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

    # 构造输入
    inputs = {
        "messages": [SystemMessage(content=system_prompt)] + history_context + [HumanMessage(content=isolated_input)]
    }
    
    # 执行 ReAct 流程
    result = await rag_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    
    return {
        "messages": [AIMessage(content=final_content, name="general_chat")],
        "task_board": [update_task_result(task, result=final_content)]
    }    