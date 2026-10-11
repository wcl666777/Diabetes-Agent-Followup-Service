"""血糖管理 Agent 的 LangGraph 图。

拓扑：START → record → interpret → END
（必要时可在 record/interpret 之间加条件边做告警分流，当前直接串联。）
"""
from langgraph.graph import END, START, StateGraph

from app.agents.glucose_agent.nodes import interpret_node, record_node
from app.agents.glucose_agent.state import GlucoseState
from app.core.memory import get_memory_saver


def build_glucose_graph():
    builder = StateGraph(GlucoseState)

    builder.add_node("record", record_node)
    builder.add_node("interpret", interpret_node)

    builder.add_edge(START, "record")
    builder.add_edge("record", "interpret")
    builder.add_edge("interpret", END)

    checkpointer = get_memory_saver("glucose")
    return builder.compile(checkpointer=checkpointer)