"""
Day 5: 把图结构画出来
"""

import sys
sys.path.insert(0, "..")
from complete_agent import build_complete_agent

app = build_complete_agent()

# 打印 ASCII 流程图
print("你的 Agent 图结构:\n")
print(app.get_graph().draw_ascii())