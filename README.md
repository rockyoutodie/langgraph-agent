# 🕸️ LangGraph Agent — 基于状态图的智能助理

> 用 LangGraph 构建的图式 Agent,具备多分支路由、ReAct 推理、对话记忆与持久化。相比手写 Agent,用"状态图"让复杂流程清晰可维护。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-green)
![Ollama](https://img.shields.io/badge/Ollama-Local-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ 核心特性

- 🧭 **多分支路由**: 自动判断问题类型,简单闲聊快速回答,复杂任务走完整 ReAct
- 🔄 **ReAct 推理循环**: 思考→行动→观察,自动分解多步任务
- 🧠 **对话记忆**: 基于 Checkpoint,自动维护上下文,理解"刚才""上面的结果"等指代
- 💾 **持久化支持**: 可选 SQLite 持久化,记忆跨程序重启不丢
- 👥 **多会话隔离**: 用 thread_id 隔离不同用户,记忆互不干扰
- 📊 **结构可视化**: 图结构可导出,一目了然

---

## 🏗️ 架构

\`\`\`
              START
                ↓
           ┌────────┐
           │ route  │  判断问题类型
           └────────┘
                ↓
          <chat 还是 agent?>
       ┌────────┴────────┐
       ↓                 ↓
   [chat_node]      [call_llm] ←──────┐
       ↓                ↓             │
      END          <要工具吗?>         │
                    ├─ 要 → [execute_tools]
                    └─ 不要 → END
\`\`\`

整张图挂载 Checkpointer,自动实现记忆与持久化。

### 节点职责

| 节点          | 职责                    |
| ------------- | ----------------------- |
| route         | 判断问题类型,分发路径   |
| chat          | 处理纯闲聊,直接回答     |
| call_llm      | 调用 LLM,决定是否用工具 |
| execute_tools | 执行工具调用            |

---

## 🤔 设计权衡：路由式架构的取舍

本项目采用"路由 + 分支"架构,但这带来一个**真实的工程权衡**:

**优势**: 简单问题走 chat 快速回答,不浪费资源;复杂问题走完整 ReAct。

**风险**: 路由判断依赖 LLM,**一旦路由错误,请求会被送到错误路径且无法补救**。例如"把上面的结果除以2"若被误判为 chat,就会失去调用计算器的机会。

**本项目的应对**: 路由策略**偏向 agent**——拿不准时优先走全能的 agent 路径(既能聊天也能用工具),把"送错路"的风险降到最低。这是"效率"与"可靠性"之间的折中。

**其他可选方案**:
- 不做路由,全走 ReAct(最可靠,但简单问题也慢)
- 用关键词规则 + LLM 混合路由(兼顾快与准)

> 没有完美的架构,每个设计都是权衡。选择取决于具体场景对"速度"和"可靠性"的偏好。

---

## 🚀 快速开始

### 前置要求
- Python 3.10+
- [Ollama](https://ollama.com) + \`ollama pull qwen2.5:14b\`

### 安装
\`\`\`bash
git clone https://github.com/rockyoutodie/langgraph-agent.git
cd langgraph-agent
pip install -r requirements.txt
\`\`\`

### 运行
\`\`\`bash
python3 complete_agent.py    # 交互式对话
\`\`\`

试试:
\`\`\`
你好                          → 走 chat
北京天气怎么样?                → 走 agent,查天气
我刚才问的哪个城市?             → 记忆生效
50 美元换人民币再乘以 3         → 走 agent,多步 ReAct
把上面的结果除以 2             → 记忆 + 计算
\`\`\`

---

## 📂 项目结构

\`\`\`
langgraph-agent/
├── complete_agent.py   # 主项目: 路由 + ReAct + 记忆
├── tools.py            # 工具集
├── langgraph_agent.py  # 基础 ReAct 图
├── router_agent.py     # 多分支路由
├── memory_agent.py     # Checkpoint 记忆
├── experiments/        # 学习实验
└── README.md
\`\`\`

---

## 🧠 LangGraph 核心概念

| 概念             | 说明                   | 对应                   |
| ---------------- | ---------------------- | ---------------------- |
| State            | 贯穿流程的共享数据     | 手写 Agent 的 messages |
| Node             | 流程中的一步(一个函数) | 调 LLM / 执行工具      |
| Edge             | 节点间的流转           | 普通边 / 条件边        |
| Conditional Edge | 根据条件分发           | 手写的 if/else         |
| Checkpoint       | 自动存档 State         | 记忆 + 持久化          |
| thread_id        | 会话标识               | 多用户隔离             |

---

## 🎓 技术栈

- **LangGraph** — 状态图 Agent 框架
- **qwen2.5:14b** — 决策 + 生成
- **Ollama** — 本地推理
- **Checkpoint (MemorySaver / SqliteSaver)** — 记忆与持久化

---

## 📝 学到的核心知识

- ✅ 状态图建模: 用 State/Node/Edge 描述 Agent 流程
- ✅ 条件边实现路由与循环
- ✅ Checkpoint 实现记忆、持久化、多会话隔离
- ✅ 路由式架构的效率/可靠性权衡
- ✅ 手写 Agent vs 框架 Agent 的取舍

---

## 🔮 未来改进

- [ ] 混合路由(关键词 + LLM),降低路由错误率
- [ ] 接入更多工具(搜索、RAG、文件操作)
- [ ] 用专用 reranker 和更强模型提升稳定性
- [ ] 完整的 SQLite/Postgres 持久化

---

## 📝 License

MIT

---

## 🙋 关于作者

前端开发者转型 AI 应用工程师。这是我的第五个 AI 项目,从手写 Agent 进阶到 LangGraph 框架。

**前四个项目**: 会议纪要 / 文本工具箱 / RAG 问答 / 手写 Agent
**本项目**: 用 LangGraph 重构,掌握工业级 Agent 框架