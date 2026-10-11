"""问答流中「患者主动要求人工」的公共实现。

患者明确要求人工（如「我要找医生」「转人工」）时，跳过检索让问答图直接 interrupt，
登记待处理记录并返回 (thread_id, reason)，供 chat 与 unified_chat 两个入口复用。
"""
from langchain_core.messages import HumanMessage

from app.agents.qa_agent.graph import build_qa_graph
from app.core.logger import get_logger
from app.core.memory import build_thread_id
from app.core.tracking import track_pending

logger = get_logger(__name__)


def _unwrap(payload) -> dict:
    """兼容不同 LangGraph 版本对 __interrupt__ 的包装，取出 interrupt(value)。"""
    if isinstance(payload, (tuple, list)):
        for item in payload:
            value = getattr(item, "value", item)
            return value if isinstance(value, dict) else {"reason": str(value)}
    if isinstance(payload, dict):
        return payload
    return {}


async def force_human_interrupt(
    patient_id: str, session_id: str, question: str, reason_hint: str
) -> tuple[str, str]:
    """跳过检索，直接让问答图进入人工介入并登记待处理。返回 (thread_id, reason)。"""
    thread_id = build_thread_id(patient_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_qa_graph()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=question)],
            "patient_id": patient_id,
            "session_id": session_id,
            "question": question,
            "confidence": 0.0,
            "force_human": True,
        },
        config=config,
    )
    payload = _unwrap(result.get("__interrupt__")) if result.get("__interrupt__") else {}
    reason = payload.get("reason", reason_hint)
    await track_pending(thread_id, patient_id, session_id, question, payload)
    logger.info("qa.force_human | patient=%s | reason=%s", patient_id, reason)
    return thread_id, reason
