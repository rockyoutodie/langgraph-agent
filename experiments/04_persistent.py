"""
Day 4: 持久化到 SQLite - 程序重启也不丢记忆
"""

import sys, json
sys.path.insert(0, "..")
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver   # ← SQLite 存档器
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def to_openai(messages):
    r = []
    for m in messages:
        if isinstance(m, SystemMessage): r.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage): r.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage): r.append({"role": "assistant", "content": m.content or ""})
    return r


def call_llm(state):
    resp = client.chat.completions.create(
        model="qwen2.5:14b", messages=to_openai(state["messages"]), temperature=0.3
    )
    return {"messages": [AIMessage(content=resp.choices[0].message.content)]}


# 用 SQLite 存档 (存到磁盘文件)
with SqliteSaver.from_conn_string("agent_memory.db") as checkpointer:
    graph = StateGraph(State)
    graph.add_node("chat", call_llm)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "persistent_test"}}
    
    # 检查是否有历史记忆
    existing = app.get_state(config)
    if existing.values.get("messages"):
        print("📂 发现之前的对话记忆:")
        for m in existing.values["messages"]:
            role = "你" if isinstance(m, HumanMessage) else "AI"
            print(f"   [{role}] {m.content[:50]}")
        print()
        # 基于记忆继续
        user_input = "我们刚才聊到哪了?"
    else:
        print("🆕 全新对话,没有历史记忆\n")
        user_input = "你好,请记住我最喜欢的数字是 42"
    
    print(f"👤 你: {user_input}")
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    print(f"🤖 {result['messages'][-1].content}")
    
    print(f"\n💡 现在退出,再跑一次这个脚本,它会记得这次对话!")
    print(f"   记忆存在: agent_memory.db")