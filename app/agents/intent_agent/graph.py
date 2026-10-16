"""意图识别 Agent 的 LangGraph 图。

拓扑：START → classify → END

单节点图。之所以仍做成独立的 LangGraph 图（而不是普通函数），
是为了让它能被编排器当作「一个 Agent」统一调度，
并作为「意图识别 Agent → 问答 Agent」两步编排的第一环。
"""
from langgraph.graph import END, START, StateGraph

from app.agents.intent_agent.nodes import classify_node
from app.agents.intent_agent.state import IntentState
from app.core.memory import get_memory_saver


def build_intent_graph():
    builder = StateGraph(IntentState)

    builder.add_node("classify", classify_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)

    checkpointer = get_memory_saver("intent")
    return builder.compile(checkpointer=checkpointer)
