"""问答 Agent 的 LangGraph 图。

拓扑：
    START → retrieve → (路由) → generate          (高置信且非高危)
                              → human_review(interrupt) → END   (低置信 / 高危 / 主动要求人工)
    generate → END

说明：医生接管时直接以医生输入作为回答（human_review → END），不再调用大模型生成。
"""
from langgraph.graph import END, START, StateGraph

from app.agents.qa_agent.nodes import (
    generate_node,
    human_review_node,
    retrieve_node,
    _route_after_retrieve,
)
from app.agents.qa_agent.state import QAState
from app.core.memory import get_memory_saver


def build_qa_graph():
    builder = StateGraph(QAState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")

    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {
            "generate": "generate",
            "human_review": "human_review",
        },
    )
    builder.add_edge("human_review", END)
    builder.add_edge("generate", END)

    checkpointer = get_memory_saver("qa")
    return builder.compile(checkpointer=checkpointer)