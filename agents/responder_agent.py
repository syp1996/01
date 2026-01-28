'''
Author: Yunpeng Shi y.shi27@newcastle.ac.uk
FilePath: /01/agents/responder_agent.py
Description: 并行化改造版 - 修复引用标记丢失问题 (Responder 增强)
'''
from langchain_core.messages import AIMessage, SystemMessage

from state import agentState
from utils import llm


async def responder_agent(state: agentState):
    board = state.get("task_board", [])
    
    # 1. 拼接上下文 (增加显眼的分割线，帮助模型识别引用块)
    results_context = "【⬇️ 下面是各部门提交的执行结果 (包含重要引用信息) ⬇️】:\n"
    for i, task in enumerate(board):
        res = task.get("result", "无结果")
        # 这里的 f-string 会把 Worker 生成的带标记的文本拼进来
        results_context += f">>> 任务 {i+1} ({task.get('task_type')}): {task.get('description', '')}\n执行结果: {res}\n\n"

    print("\n[Responder] 正在汇总结果 (已开启引用保护模式)...")

    # 2. 核心 Prompt 修改
    system_prompt = """
    你是杭州地铁的资深智能服务向导。
    你的任务是根据【各部门提交的执行结果】，为用户生成一段温暖、通顺的最终答复。

    ### ⚠️ 最高优先级原则 (引用保护)：
    **绝对禁止删除引用标记！** 如果在子任务结果中出现了 `【📚知识库】` 或 `(来源:📚知识库)` 等标记，**你必须将其原样保留在你的最终回复中**。
    - ❌ 错误做法：为了语句通顺，把标记删掉了。
    - ✅ 正确做法：...根据规定，折叠自行车可以携带(来源:📚知识库)。另外提醒您...

    ### 其他要求：
    1. **数据忠实**：票价、时间、尺寸数字，必须与子任务结果完全一致，一个字都不能改。
    2. **拟人化包装**：使用“为您查询到了”、“专门帮您确认了”等连接词，让对话有温度。
    3. **逻辑整合**：不要机械罗列任务1、任务2，要将它们融合成一段完整的话。
    """
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    messages.append(SystemMessage(content=results_context))

    response = await llm.ainvoke(messages)
    return {"messages": [AIMessage(content=response.content, name="responder_agent")]}