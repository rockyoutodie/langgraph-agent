"""
Day 3: 多分支路由 Agent
先判断问题类型,再走不同的处理路径
"""

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from openai import OpenAI

from tools import TOOLS_SCHEMA, AVAILABLE_FUNCTIONS

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ============================================================
# State: 多了一个 route_type 字段记录路由结果
# ============================================================

class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    route_type: str    # 路由类型: chat / query / complex


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
# 节点 1: 路由节点 (判断问题类型)
# ============================================================

def route_node(state: RouterState) -> dict:
    """用 LLM 判断问题类型"""
    print("  🧭 [节点] route - 判断问题类型")
    
    # 取最后一条用户消息
    user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_msg = m.content
            break
    
    # 让 LLM 分类
    classify_prompt = f"""判断下面用户问题的类型,只回答一个词:

- chat: 闲聊、打招呼、问你是谁等,不需要任何工具
- query: 简单的单次查询,如查天气、查汇率、单个计算
- complex: 复杂的多步任务,需要多个工具配合,如"换汇率再计算"

用户问题: {user_msg}

只回答 chat、query 或 complex 中的一个词:"""

    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": classify_prompt}],
        temperature=0,
    )
    
    route_type = response.choices[0].message.content.strip().lower()
    # 容错: 只保留合法值
    if "complex" in route_type:
        route_type = "complex"
    elif "query" in route_type:
        route_type = "query"
    else:
        route_type = "chat"
    
    print(f"     判定类型: {route_type}")
    return {"route_type": route_type}


# ============================================================
# 节点 2: 闲聊处理 (直接回答,不用工具)
# ============================================================

def chat_node(state: RouterState) -> dict:
    print("  💬 [节点] chat - 直接回答")
    
    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=to_openai_messages(state["messages"]),
        temperature=0.7,   # 闲聊可以活泼点
    )
    answer = response.choices[0].message.content
    return {"messages": [AIMessage(content=answer)]}


# ============================================================
# 节点 3: 查询处理 (单次工具调用)
# ============================================================

def query_node(state: RouterState) -> dict:
    print("  🔍 [节点] query - 单次工具调用")
    
    # 第一次: LLM 决定用什么工具
    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=to_openai_messages(state["messages"]),
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0,
    )
    msg = response.choices[0].message
    
    if not msg.tool_calls:
        return {"messages": [AIMessage(content=msg.content or "")]}
    
    # 执行工具
    tc = msg.tool_calls[0]
    name = tc.function.name
    args = json.loads(tc.function.arguments)
    print(f"     执行 {name}({args})")
    result = AVAILABLE_FUNCTIONS[name](**args) if name in AVAILABLE_FUNCTIONS else "未知工具"
    print(f"     结果: {result}")
    
    # 第二次: 组织答案
    msgs = to_openai_messages(state["messages"])
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}
    ]})
    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    
    final = client.chat.completions.create(model="qwen2.5:14b", messages=msgs, temperature=0)
    return {"messages": [AIMessage(content=final.choices[0].message.content)]}


# ============================================================
# 节点 4: 复杂任务处理 (完整 ReAct - 内部循环)
# ============================================================

def complex_node(state: RouterState) -> dict:
    print("  🧩 [节点] complex - 完整 ReAct")
    
    msgs = to_openai_messages(state["messages"])
    
    # 内部 ReAct 循环 (最多 8 步)
    for step in range(8):
        response = client.chat.completions.create(
            model="qwen2.5:14b",
            messages=msgs,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        
        if not msg.tool_calls:
            return {"messages": [AIMessage(content=msg.content or "")]}
        
        # 记录 assistant 消息
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})
        
        # 执行工具
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"     执行 {name}({args})")
            result = AVAILABLE_FUNCTIONS[name](**args) if name in AVAILABLE_FUNCTIONS else "未知工具"
            print(f"     结果: {result}")
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    
    return {"messages": [AIMessage(content="任务过于复杂,未能完成")]}


# ============================================================
# 条件函数: 根据 route_type 决定去哪个节点
# ============================================================

def route_decision(state: RouterState) -> str:
    """返回下一个节点的名字"""
    rt = state["route_type"]
    print(f"  ➡️  [条件边] 路由到: {rt}")
    return rt


# ============================================================
# 构建图 (对照开头那张图!)
# ============================================================

def build_router_agent():
    graph = StateGraph(RouterState)
    
    # 添加所有节点
    graph.add_node("route", route_node)
    graph.add_node("chat", chat_node)
    graph.add_node("query", query_node)
    graph.add_node("complex", complex_node)
    
    # START → route
    graph.add_edge(START, "route")
    
    # 条件边: route 之后,根据类型分发到 3 条路径
    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "chat": "chat",
            "query": "query",
            "complex": "complex",
        }
    )
    
    # 三条路径都通向 END
    graph.add_edge("chat", END)
    graph.add_edge("query", END)
    graph.add_edge("complex", END)
    
    return graph.compile()


SYSTEM_PROMPT = """你是一个能干的助理,可以计算、查天气、查汇率。
数学运算必须用 calculator,并代入实际数字(如 '359 * 3',不要用变量名)。用中文回答。"""


def run(user_message: str):
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_message}")
    print(f"{'='*60}")
    
    app = build_router_agent()
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ],
        "route_type": "",
    }
    final_state = app.invoke(initial_state, {"recursion_limit": 20})
    answer = final_state["messages"][-1].content
    print(f"\n💬 最终答案: {answer}")
    return answer


if __name__ == "__main__":
    tests = [
        "你好,你是谁?",                          # → chat
        "北京天气怎么样?",                        # → query
        "我有 50 美元,换成人民币再乘以 3 是多少?",  # → complex
    ]
    for t in tests:
        run(t)