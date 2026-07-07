"""
Day 1 实验 2: 条件边 - 图的精华
根据数字正负,走不同的节点
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    number: int
    result: str


def check_number(state: State) -> dict:
    """入口节点: 只是记录一下"""
    print(f"  检查数字: {state['number']}")
    return {}


def handle_positive(state: State) -> dict:
    print("  → 走了'正数'分支")
    return {"result": f"{state['number']} 是正数"}


def handle_negative(state: State) -> dict:
    print("  → 走了'负数'分支")
    return {"result": f"{state['number']} 是负数或零"}


# ============================================================
# 条件函数: 决定走哪条边
# ============================================================

def route(state: State) -> str:
    """根据数字正负,返回下一个节点的名字"""
    if state["number"] > 0:
        return "positive"   # 返回节点名
    else:
        return "negative"


# ============================================================
# 构建带条件边的图
# ============================================================

graph = StateGraph(State)

graph.add_node("check", check_number)
graph.add_node("positive", handle_positive)
graph.add_node("negative", handle_negative)

graph.add_edge(START, "check")

# 条件边: check 之后,根据 route 函数决定去 positive 还是 negative
graph.add_conditional_edges(
    "check",           # 从哪个节点出发
    route,             # 用哪个函数判断
    {
        "positive": "positive",   # route 返回 "positive" → 去 positive 节点
        "negative": "negative",   # route 返回 "negative" → 去 negative 节点
    }
)

graph.add_edge("positive", END)
graph.add_edge("negative", END)

app = graph.compile()


if __name__ == "__main__":
    for num in [5, -3, 0]:
        print(f"\n{'='*40}")
        print(f"输入: {num}")
        result = app.invoke({"number": num, "result": ""})
        print(f"结果: {result['result']}")