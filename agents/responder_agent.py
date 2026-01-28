'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/responder_agent.py
Description: 并行化改造版 - 负责汇总所有 Worker 的结果并回复用户
'''
from langchain_core.messages import AIMessage, SystemMessage

from state import agentState
from utils import llm  # 移除了 broken 的 complete_current_task 引用


async def responder_agent(state: agentState):
    """
    Responder 节点：
    1. 读取全局 task_board 中的所有任务结果。
    2. 综合生成最终回复。
    注意：它接收的是 agentState (全局状态)，而不是 WorkerState。
    """
    
    # 1. 获取看板信息
    board = state.get("task_board", [])
    
    # 2. 提取任务结果
    # 过滤出已经完成的任务结果，拼接成 context
    results_context = "【子任务执行结果汇总】:\n"
    for i, task in enumerate(board):
        status = task.get("status", "unknown")
        result = task.get("result", "无结果")
        description = task.get("description", "")
        
        if status == "done":
            results_context += f"{i+1}. 任务: {description}\n   结果: {result}\n\n"
        else:
            results_context += f"{i+1}. 任务: {description}\n   状态: {status} (未完成)\n\n"

    print("\n[Responder] 正在汇总结果...")

    # 3. 构建 Prompt
    # 我们把原始对话历史 + 子任务结果 一起丢给大模型
    system_prompt = """
    你是智能客服系统的最终回复生成器。
    你的任务是根据【子任务执行结果汇总】，回答用户的原始问题。
    
    要求：
    1. 语言通顺、自然，不要机械地罗列结果。
    2. 如果子任务结果包含具体数据（如票价、时间），必须准确引用。
    3. 如果子任务未能查到信息，请如实告知用户。
    4. 忽略内部的任务ID和处理过程，只给用户他们关心的答案。
    """
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    # 将结果汇总作为最新的系统消息补充进去
    messages.append(SystemMessage(content=results_context))

    # 4. 生成回复
    response = await llm.ainvoke(messages)
    
    # 5. 返回
    # Responder 结束后通常就是 END，不需要再更新 task_board
    return {
        "messages": [AIMessage(content=response.content, name="responder_agent")]
    }