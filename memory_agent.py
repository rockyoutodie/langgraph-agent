"""
Day 4: 带记忆的图 Agent
用 LangGraph Checkpoint 实现对话记忆
"""

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver   # ← 关键: 内存存档器
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from openai import OpenAI

from tools import TOOLS_SCHEMA, AVAILABLE_FUNCTIONS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def to_openai_messages(messages):
    result = []
    for m in messages:
        if isinstance(m, SystemMessage):
            result.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msg = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                    for tc in m.tool_calls
                ]
            result.append(msg)
        elif isinstance(m, ToolMessage):
            result.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return result


def call_llm(state: AgentState) -> dict:
    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=to_openai_messages(state["messages"]),
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "args": json.loads(tc.function.arguments)}
            for tc in msg.tool_calls
        ]
        return {"messages": [AIMessage(content=msg.content or "", tool_calls=tool_calls)]}
    return {"messages": [AIMessage(content=msg.content or "")]}


def execute_tools(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_msgs = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        result = AVAILABLE_FUNCTIONS[name](**args) if name in AVAILABLE_FUNCTIONS else "未知工具"
        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": tool_msgs}


def should_continue(state: AgentState) -> str:
    return "continue" if state["messages"][-1].tool_calls else "end"


# ============================================================
# 构建图 + 加 Checkpoint (关键就这里!)
# ============================================================

def build_agent_with_memory():
    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", should_continue,
                                {"continue": "execute_tools", "end": END})
    graph.add_edge("execute_tools", "call_llm")
    
    # ★★★ 关键: 加一个 checkpointer,记忆就有了 ★★★
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


SYSTEM_PROMPT = """你是一个能干的助理,可以计算、查天气、查汇率。
数学运算用 calculator 并代入实际数字。充分利用对话历史理解"刚才""那个"等指代。用中文回答。"""


# ============================================================
# 测试: 连续对话,验证记忆
# ============================================================

if __name__ == "__main__":
    app = build_agent_with_memory()
    
    # ★★★ thread_id: 标识一个对话会话 ★★★
    config = {"configurable": {"thread_id": "user_001"}}
    
    def chat(user_input):
        print(f"\n👤 你: {user_input}")
        # 注意: 只传新消息! 历史由 checkpointer 自动补上
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        answer = result["messages"][-1].content
        print(f"🤖 {answer}")
        return answer
    
    # 第一轮需要带 system prompt
    app.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="记住,我叫小明")]},
        config=config
    )
    print("👤 你: 记住,我叫小明")
    print("🤖 (已记住)")
    
    # 后续对话 - 测试记忆!
    chat("我叫什么名字?")              # 测试: 记得名字吗?
    chat("北京天气怎么样?")            # 查询
    chat("我刚才问的是哪个城市的天气?")  # 测试: 记得刚才问了北京吗?