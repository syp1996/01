'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-27 10:57:25
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-01-28 11:00:51
FilePath: /01/agents/responder_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import agentState
from utils import complete_current_task, llm


async def responder_agent(state: agentState):
    print("   [Responder] 所有任务已完成，正在生成最终回复...")
    
    # 1. 获取所有子智能体的劳动成果
    results = state.get("task_results", {})
    
    # 2. 获取用户的原始问题（用于给 LLM 提供上下文）
    # 我们倒序查找最后一条用户消息
    original_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            original_input = msg.content
            break

    # 3. 拼接上下文给 LLM
    context_str = "\n".join([f"【{k}的处理结果】: {v}" for k, v in results.items()])
    
    # 4. 编写 Prompt
    prompt = f"""
    你是一名**资深且热心**的地铁客服经理。
    你的服务宗旨是：**提供超出用户预期的详细解答，绝对不敷衍。**
    
    用户的原始问题是："{original_input}"
    
    你的各部门同事已经给出了处理结果，请你汇总这些信息，给用户一个**连贯、亲切、结构清晰**的最终回复。
    
    【各部门处理结果】：
    {context_str}
    
    【要求】：
    1. **扩写与润色（关键）**：不要只是简单拼接素材。请对每一条信息进行**展开说明**。
       - 如果是禁止的，请温和地解释原因（如：为了安全、为了秩序）。
       - 如果是允许的，请补充一些温馨提示（如：折叠好、注意尺寸）。
    2. **保留来源标记**：素材中出现的 `【📚知识库】` 标记必须保留在对应句子的末尾，作为权威凭证。
    3. **增加关怀语**：在回复的开头和结尾，加入更有温度的寒暄和关怀（如天气提醒、出行安全提示），让回复看起来不像机器人。
    4. **结构清晰**：内容较多时，请使用分点陈述。
    请生成一份**字数适中偏长、语气温暖**的完整回复。
    """
    
    # 5. 调用模型生成最终回复
    final_response = await llm.ainvoke([SystemMessage(content=prompt)])
    
    # 6. 返回结果，并顺便清空看板和结果池（为下一轮对话重置状态）
    return {
        "messages": [AIMessage(content=final_response.content, name="final_responder")],
        "task_board": [],     # 清空看板
        "task_results": {}    # 清空结果池
    }
