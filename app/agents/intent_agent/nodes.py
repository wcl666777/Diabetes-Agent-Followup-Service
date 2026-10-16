"""意图识别 Agent 的节点实现。

策略：LLM 主导（理解口语化表达）→ 规则兜底（模型异常/输出无法解析时保证有结果）。
与 RAG 侧 query_classifier（通用知识 / 专业咨询）分工不同：
  本 Agent 决定「答题往哪个方向侧重」，RAG 侧决定「要不要检索、走哪种检索策略」。
"""
import json
import re

from langchain_core.messages import AIMessage

from app.agents.intent_agent.prompts import INTENT_CLASSIFY_PROMPT
from app.agents.intent_agent.state import IntentState
from app.core.llm_factory import get_llm
from app.core.logger import get_logger
from app.core.question_types import (
    DEFAULT_QUESTION_TYPE,
    QUESTION_TYPES,
    classify_question_type,
)

logger = get_logger(__name__)

_VALID_TYPES = set(QUESTION_TYPES)


def _parse_type(text: str) -> tuple[str, float]:
    """从模型输出中解析 (类型, 置信度)；解析失败返回 ("", 0.0)。"""
    matched = re.search(r"\{.*\}", text, re.DOTALL)
    if matched:
        try:
            obj = json.loads(matched.group(0))
            qtype = str(obj.get("question_type", "")).strip()
            confidence = float(obj.get("confidence", 0.0) or 0.0)
            if qtype in _VALID_TYPES:
                return qtype, confidence
        except (ValueError, TypeError):
            pass
    # 兼容模型只回裸类型名（未按 JSON 输出）：按 QUESTION_TYPES 顺序匹配，结果稳定
    for qtype in QUESTION_TYPES:
        if qtype in text:
            return qtype, 0.5
    return "", 0.0


async def classify_intent(question: str) -> tuple[str, float, str]:
    """核心分类逻辑（供图节点与编排器直接调用），返回 (类型, 置信度, 来源)。

    异常一律不外抛：失败时退回规则分类，来源标记为 rule，
    保证调用方永远拿得到一个可用的类型。
    """
    question = (question or "").strip()
    if not question:
        return DEFAULT_QUESTION_TYPE, 0.0, "rule"

    try:
        llm = get_llm("intent", temperature=0)
        resp = await llm.ainvoke(INTENT_CLASSIFY_PROMPT.format_messages(question=question))
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        qtype, confidence = _parse_type(text)
        if qtype:
            logger.info(
                "intent_agent.llm_ok | type=%s | conf=%.2f | q=%s",
                qtype, confidence, question[:30],
            )
            return qtype, confidence, "llm"
        logger.warning("intent_agent.llm_unparsed | text=%s", text[:80])
    except Exception as e:
        logger.warning("intent_agent.llm_failed | error=%s", e)

    qtype = classify_question_type(question)
    logger.info("intent_agent.rule_fallback | type=%s | q=%s", qtype, question[:30])
    return qtype, 0.0, "rule"


async def classify_node(state: IntentState) -> dict:
    """LangGraph 节点：识别问题类型并写入 state（结构化输出供下游 Agent 使用）。"""
    qtype, confidence, source = await classify_intent(state.get("question") or "")
    return {
        "question_type": qtype,
        "confidence": confidence,
        "source": source,
        "structured_output": {
            "question_type": qtype,
            "confidence": confidence,
            "source": source,
        },
        "messages": [AIMessage(content=qtype)],
    }
