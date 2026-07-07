"""
用 LangGraph 重建 ReAct Agent (LangChain 消息对象版 - 更规范)
"""

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from openai import OpenAI

from tools import TOOLS_SCHEMA, AVAILABLE_FUNCTIONS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ============================================================
# 1. State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 辅助: 把 LangChain 消息对象转成 OpenAI 格式
# ============================================================

def to_openai_messages(messages):
    """LangChain 消息对象 → OpenAI API 格式的 dict"""
    result = []
    for m in messages:
        if isinstance(m, SystemMessage):
            result.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msg = {"role": "assistant", "content": m.content or ""}
            # 如果有工具调用,带上
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}
                    }
                    for tc in m.tool_calls
                ]
            result.append(msg)
        elif isinstance(m, ToolMessage):
            result.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
    return result


# ============================================================
# 2. 节点 A: 调用 LLM
# ============================================================

def call_llm(state: AgentState) -> dict:
    print("  🧠 [节点] call_llm")
    
    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=to_openai_messages(state["messages"]),
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0,
    )
    
    msg = response.choices[0].message
    
    # 构造 AIMessage 对象
    if msg.tool_calls:
        # 有工具调用
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
            }
            for tc in msg.tool_calls
        ]
        ai_msg = AIMessage(content=msg.content or "", tool_calls=tool_calls)
    else:
        ai_msg = AIMessage(content=msg.content or "")
    
    return {"messages": [ai_msg]}


# ============================================================
# 3. 节点 B: 执行工具
# ============================================================

def execute_tools(state: AgentState) -> dict:
    print("  🔧 [节点] execute_tools")
    
    last_message = state["messages"][-1]   # 这是 AIMessage 对象
    
    tool_messages = []
    for tc in last_message.tool_calls:      # 用 .tool_calls 属性
        name = tc["name"]
        args = tc["args"]
        
        print(f"     执行 {name}({args})")
        
        if name in AVAILABLE_FUNCTIONS:
            result = AVAILABLE_FUNCTIONS[name](**args)
        else:
            result = f"未知工具: {name}"
        
        print(f"     结果: {result}")
        
        # 构造 ToolMessage 对象
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    
    return {"messages": tool_messages}


# ============================================================
# 4. 条件边: 用 .tool_calls 属性,不用 .get()
# ============================================================

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]   # AIMessage 对象
    
    # 关键修复: 用属性访问,不用 .get()
    if last_message.tool_calls:
        print("  ➡️  [条件边] LLM 要调工具 → execute_tools")
        return "continue"
    else:
        print("  ➡️  [条件边] LLM 完成 → END")
        return "end"


# ============================================================
# 5. 构建图
# ============================================================

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {"continue": "execute_tools", "end": END}
    )
    graph.add_edge("execute_tools", "call_llm")
    return graph.compile()


# ============================================================
# 6. 使用
# ============================================================

SYSTEM_PROMPT = """你是一个能干的助理,可以计算、查天气、查汇率。

重要:
- 数学运算必须用 calculator,不要心算。
- 调用 calculator 时,必须代入实际的数字,不要使用变量名。
  例如上一步得到 359,下一步要算它乘以 3,应该写 '359 * 3',
  绝对不要写 'result * 3' 这种引用变量的写法,因为不存在这样的变量。
- 多步任务一步步来,把上一步的实际结果数值代入下一步。
用中文回答。"""


def run(user_message: str):
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_message}")
    print(f"{'='*60}")
    
    app = build_agent()
    print(app.get_graph().draw_ascii()) 
    
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    }
    
    final_state = app.invoke(initial_state, {"recursion_limit": 15})
    
    answer = final_state["messages"][-1].content   # 用 .content 属性
    print(f"\n💬 最终答案: {answer}")
    return answer


if __name__ == "__main__":
    run("我有 50 美元,换成人民币后,再乘以 3 是多少?")