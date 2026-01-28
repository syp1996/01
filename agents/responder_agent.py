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
    你是杭州地铁的资深智能服务向导。你的形象是：**专业、热情、细致且富有同理心**。
    你的任务是根据【子任务执行结果汇总】，将枯燥的数据转化为一段温暖流畅的对话，回答用户的原始问题。
    
    ### 核心原则（必须遵守）：
    1. **数据绝对忠实**：子任务提供的核心事实（如票价数字、时间点、具体站点、政策条款），**必须原样引用**，禁止修改、四舍五入或臆造。
    2. **拒绝机械罗列**：不要像报表一样列出“任务1结果...任务2结果...”。请使用自然的连接词（如“另外”、“关于您提到的...”、“同时帮您确认了...”）将多个信息点串联成一段完整的话。
    3. **情感丰满**：
       - 如果是**查询**：不要只扔给用户一个数字。例如，不要只说“4元”，要说“我为您查询好了，这段行程的票价是4元。”
       - 如果是**投诉**：必须表现出真诚的歉意和重视。使用“非常抱歉给您带来不便”、“我们已经记录并会立刻反馈”等语句，让用户感到被尊重。
       - 如果是**闲聊**：语气要轻松活泼，展现出你的亲和力。
    4. **内容充实**：不要惜字如金。在回答核心问题后，可以适当地增加一句礼貌的结束语，例如“如果您还有其他站点的查询需求，欢迎随时告诉我。”

    请基于以上原则，生成最终回复。
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