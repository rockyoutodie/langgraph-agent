"""
Day 4: thread_id 隔离多个对话
"""

import sys
sys.path.insert(0, "..")
from memory_agent import build_agent_with_memory, SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


app = build_agent_with_memory()


def chat(user_input, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    return result["messages"][-1].content


# ============================================================
# 两个用户,两个 thread_id,记忆互不干扰
# ============================================================

# 用户 A (thread: alice)
app.invoke(
    {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="我叫爱丽丝,我喜欢猫")]},
    config={"configurable": {"thread_id": "alice"}}
)

# 用户 B (thread: bob)
app.invoke(
    {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="我叫鲍勃,我喜欢狗")]},
    config={"configurable": {"thread_id": "bob"}}
)

print("=" * 50)
print("测试: 两个对话的记忆是否隔离?")
print("=" * 50)

# 问爱丽丝的对话
print(f"\n[对话 alice] 我叫什么?喜欢什么?")
print(f"🤖 {chat('我叫什么?喜欢什么动物?', 'alice')}")

# 问鲍勃的对话
print(f"\n[对话 bob] 我叫什么?喜欢什么?")
print(f"🤖 {chat('我叫什么?喜欢什么动物?', 'bob')}")

print("\n💡 观察: alice 应该记得'爱丽丝+猫', bob 应该记得'鲍勃+狗'")
print("   两个对话的记忆完全隔离,互不干扰")