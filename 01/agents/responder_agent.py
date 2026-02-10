'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
Date: 2026-01-28 15:19:55
LastEditors: Yunpeng Shi y.shi27@newcastle.ac.uk
LastEditTime: 2026-02-10 15:07:40
FilePath: /general_agent/01/agents/responder_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Yunpeng Shi
Description: 通用汇总智能体 - 负责整合多任务结果并进行润色
'''
from langchain_core.messages import AIMessage, SystemMessage
from state import agentState
from utils import llm


async def responder_agent(state: agentState):
    board = state.get("task_board", [])
    
    # 1. 拼接上下文（让汇总模型看清楚每个部门干了什么）
    results_context = "【⬇️ 任务执行报告 - 供参考 ⬇️】:\n"
    for i, task in enumerate(board):
        res = task.get("result", "未提供有效执行结果")
        results_context += f">>> [任务 {i+1}] 类型: {task.get('task_type')}\n描述: {task.get('description', '')}\n输出结果: {res}\n\n"

    # 2. 核心 Prompt 修改
    system_prompt = """
    你是**全能私人助理的首席协调官**。
    你的任务是将后台各个专业助手（如搜索、代码、知识库助手）的执行结果，整合并翻译成一段自然、流畅、专业且温暖的最终答复。

    ### 🧠 你的思考模式 (Structured Thinking)
    在输出最终回复之前，你必须展示你的逻辑推演过程：
    **严格遵守格式：Title: <标题> 换行 Content: <思考内容>**

    Title: 信息一致性检查
    Content: 我需要核对各个助手提供的数据（如时间、价格、技术参数）是否冲突。如果有冲突，以知识库或搜索助手的最新结果为准。
    
    Title: 结构化整合策略
    Content: 用户的问题包含多个维度，我将按照“核心结论 -> 详细细节 -> 后续建议”的逻辑进行组织，确保回复不显得杂乱。

    ### 🛡️ 核心准则：
    1. **引用保护 (最高优先级)**：
       - 如果子任务结果中包含 `[参考资料]`、`【📚来源】` 或 `(Source: ...)` 等标记，**绝对禁止删除或修改这些标记**。它们是信任的基石。
    
    2. **数据忠实性**：
       - 禁止修改任何具体的数字、链接、专有名词或代码片段。你只能优化语气，不能改动事实。

    3. **拟人化包装**：
       - 避免使用“任务1已完成”这种机械的表述。
       - 使用：“我为您查询到了...”、“综合来看...”、“建议您接下来可以...”等自然的过渡语。

    4. **异常处理**：
       - 如果某个子任务失败了（结果为“无结果”），请以抱歉的口吻说明原因，并基于其他成功任务的信息给出补充建议。
    """
    
    # 构建消息序列
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    messages.append(SystemMessage(content=results_context))

    response = await llm.ainvoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content, name="responder_agent")],
        # 发送重置信号，清空看板，为下一轮对话做准备
        "task_board": "RESET"
    }