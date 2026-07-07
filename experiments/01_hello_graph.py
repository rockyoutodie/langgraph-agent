"""
Day 1 实验 1: 最小 LangGraph - 理解节点和边
一个简单的两步流程: 打招呼 → 说再见
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 定义 State (贯穿流程的数据)
# ============================================================

class State(TypedDict):
    message: str      # 当前消息
    steps: list       # 记录走过的步骤


# ============================================================
# 2. 定义 Node (每一步做什么)
# ============================================================

def say_hello(state: State) -> dict:
    """节点 1: 打招呼"""
    print("  执行节点: say_hello")
    return {
        "message": state["message"] + " → 你好!",
        "steps": state["steps"] + ["hello"]
    }


def say_goodbye(state: State) -> dict:
    """节点 2: 说再见"""
    print("  执行节点: say_goodbye")
    return {
        "message": state["message"] + " → 再见!",
        "steps": state["steps"] + ["goodbye"]
    }


# ============================================================
# 3. 构建图
# ============================================================

# 创建图
graph = StateGraph(State)

# 添加节点
graph.add_node("hello", say_hello)
graph.add_node("goodbye", say_goodbye)

# 添加边: START → hello → goodbye → END
graph.add_edge(START, "hello")      # 开始 → 打招呼
graph.add_edge("hello", "goodbye")  # 打招呼 → 说再见
graph.add_edge("goodbye", END)      # 说再见 → 结束

# 编译成可执行的 app
app = graph.compile()


# ============================================================
# 4. 运行
# ============================================================

if __name__ == "__main__":
    print("开始运行图...")
    
    result = app.invoke({
        "message": "[开始]",
        "steps": []
    })
    
    print(f"\n最终 message: {result['message']}")
    print(f"走过的步骤: {result['steps']}")