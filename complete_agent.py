"""
第 5 周整合项目: 完整图 Agent
路由 + ReAct + 记忆 + 持久化
"""

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from openai import OpenAI

from tools import TOOLS_SCHEMA, AVAILABLE_FUNCTIONS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ============================================================
# State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route_type: str


# ============================================================
# 消息格式转换
# ============================================================

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


# ============================================================
# 节点 1: 路由 (判断走 chat 还是 agent)
# ============================================================

def route_node(state: AgentState) -> dict:
    # 找最后一条用户消息
    user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_msg = m.content
            break
    
    classify_prompt = f"""判断用户问题该走哪条路径。

- chat: 纯粹的闲聊、问候、道别、常识问答,完全不涉及任何计算/查询
- agent: 只要涉及任何计算、数字、查天气、查汇率,或对之前结果做进一步处理,都归这里

重要规则:
- 如果拿不准,一律选 agent(它更全能,既能聊天也能用工具)
- 只要出现"算""除以""乘以""加""减""换算"等词,或涉及数字处理,必须选 agent
- 只有确定是纯闲聊(你好、谢谢、再见、你是谁)才选 chat

用户问题: {user_msg}
只回答 chat 或 agent:"""

    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": classify_prompt}],
        temperature=0,
    )
    rt = response.choices[0].message.content.strip().lower()
    route_type = "agent" if "agent" in rt else "chat"
    print(f"  🧭 路由: {route_type}")
    return {"route_type": route_type}


# ============================================================
# 节点 2: 闲聊 (直接答)
# ============================================================

def chat_node(state: AgentState) -> dict:
    print("  💬 chat_node")
    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=to_openai_messages(state["messages"]),
        temperature=0.7,
    )
    return {"messages": [AIMessage(content=response.choices[0].message.content)]}


# ============================================================
# 节点 3: LLM (agent 路径的核心)
# ============================================================

def call_llm(state: AgentState) -> dict:
    print("  🧠 call_llm")
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

     # 兜底: 工具调用跑进了 content
    import re
    content = msg.content or ""
    match = re.search(r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{.*?\})\}', content, re.DOTALL)
    if match:
        try:
            name = match.group(1)
            args = json.loads(match.group(2))
            print(f"     [兜底] 从 content 捞出工具调用: {name}")
            # 构造成正规的 tool_call
            tool_calls = [{"id": "fallback_1", "name": name, "args": args}]
            return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
        except:
            pass
    
    # 真的是普通回答
    return {"messages": [AIMessage(content=content)]}



# ============================================================
# 节点 4: 执行工具
# ============================================================

def execute_tools(state: AgentState) -> dict:
    print("  🔧 execute_tools")
    last = state["messages"][-1]
    tool_msgs = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        print(f"     {name}({args})")
        result = AVAILABLE_FUNCTIONS[name](**args) if name in AVAILABLE_FUNCTIONS else "未知工具"
        print(f"     → {result}")
        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": tool_msgs}


# ============================================================
# 条件函数
# ============================================================

def route_decision(state: AgentState) -> str:
    return state["route_type"]


def should_continue(state: AgentState) -> str:
    return "continue" if state["messages"][-1].tool_calls else "end"


# ============================================================
# 构建完整图
# ============================================================

def build_complete_agent(persist=False):
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("chat", chat_node)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", route_decision, {
        "chat": "chat", "agent": "call_llm",
    })
    graph.add_edge("chat", END)
    graph.add_conditional_edges("call_llm", should_continue, {
        "continue": "execute_tools", "end": END,
    })
    graph.add_edge("execute_tools", "call_llm")
    
    # 根据参数选择记忆方式
    if persist:
        from langgraph.checkpoint.sqlite import SqliteSaver
        # 注意: SqliteSaver 需要用 context manager,这里简化用内存
        # 生产环境用 SqliteSaver.from_conn_string(...)
        checkpointer = MemorySaver()  # 演示用,持久化见 experiments/04
    else:
        checkpointer = MemorySaver()
    
    return graph.compile(checkpointer=checkpointer)


SYSTEM_PROMPT = """你是一个能干的智能助理,可以计算、查天气、查汇率。
数学运算必须用 calculator 并代入实际数字(如 '359 * 3',不要用变量名)。
充分利用对话历史理解"刚才""那个"等指代。用中文回答。"""


# ============================================================
# 交互式命令行
# ============================================================

def main():
    print("=" * 60)
    print("🤖 完整图 Agent (路由 + ReAct + 记忆)")
    print("   命令: quit 退出")
    print("=" * 60)
    
    app = build_complete_agent()
    config = {"configurable": {"thread_id": "main_session"}}
    
    # 初始化 system prompt
    first = True
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见!")
            break
        
        # 第一轮带 system prompt
        if first:
            msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]
            first = False
        else:
            msgs = [HumanMessage(content=user_input)]
        
        result = app.invoke({"messages": msgs, "route_type": ""}, config=config)
        print(f"\n🤖 {result['messages'][-1].content}")


if __name__ == "__main__":
    main()