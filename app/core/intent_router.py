"""Agent 路由意图识别（三层）。

定位：只回答「这句话该交给哪个 Agent」。它位于检索之前，因此只影响
路由准确率与调用成本，不改变检索召回率（召回率由检索层决定）。

三层结构：
  ① 规则强特征层：高精度、零成本，命中即定（如明确的「记录血糖」意图）；
  ② LLM 意图分类层：处理口语化 / 模糊表达，输出 label + confidence + reason；
  ③ 兜底层：任何异常或非法结果一律回退 qa —— 医疗场景下「问答」是最安全的兜底。

与 RAG 侧 query_classifier（通用知识 / 专业咨询）的分工：
  本模块决定「交给哪个 Agent」，RAG 侧决定「要不要检索、走哪种检索策略」，两者互不重叠。
"""
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.core.llm_factory import get_llm
from app.core.logger import get_logger

logger = get_logger(__name__)


class RouteLabel(str, Enum):
    """可路由到的 Agent 标签。"""

    QA = "qa"            # 问答 Agent（医疗咨询）
    GLUCOSE = "glucose"  # 血糖管理 Agent（记录血糖）
    CLARIFY = "clarify"  # 意图不明确，需向用户追问


@dataclass
class RouteDecision:
    """一次路由决策的结果。"""

    label: str
    reason: str
    confidence: float
    source: str  # rule / llm / fallback


# ── ① 规则强特征层 ──
# 血糖记录意图：明确的「记录动作」+ 血糖语境，精度高、零成本。
_GLUCOSE_RECORD_VERBS = ("记录", "记一下", "记下", "登记", "上报", "录入", "打卡", "存一下", "帮我记")
# 纯数值（可带单位）：如「6.5」「6.5 mmol/L」，视为上报一次血糖
_PURE_VALUE_RE = re.compile(r"^\s*\d{1,2}(\.\d+)?\s*(mmol/?l?)?\s*$", re.IGNORECASE)


def _rule_route(message: str) -> Optional[RouteDecision]:
    """规则层：仅在强特征命中时返回决策，否则返回 None 交给下一层。"""
    text = message.strip()
    if "血糖" in text and any(v in text for v in _GLUCOSE_RECORD_VERBS):
        return RouteDecision(RouteLabel.GLUCOSE.value, "命中「记录血糖」规则", 1.0, "rule")
    if _PURE_VALUE_RE.match(text):
        return RouteDecision(RouteLabel.GLUCOSE.value, "纯血糖数值，视为上报记录", 0.9, "rule")
    return None


# ── ② LLM 意图分类层 ──
_ROUTE_PROMPT = """你是糖尿病随访助手的意图路由器。请判断用户消息应交给哪个 Agent，只返回 JSON，不要多余内容。

可选 label：
- "qa"：医疗健康咨询（症状、用药、饮食、血糖指标解读、并发症等）
- "glucose"：用户想记录 / 上报一次血糖数值
- "clarify"：意图不明确，需要向用户追问

用户消息：{message}

返回格式：{{"label": "qa", "confidence": 0.9, "reason": "简短理由"}}"""

_VALID_LABELS = {label.value for label in RouteLabel}


def _llm_route(message: str) -> RouteDecision:
    """LLM 层：结构化输出 label / confidence / reason；无法解析则抛异常交上层兜底。"""
    llm = get_llm("intent", temperature=0)
    resp = llm.invoke(_ROUTE_PROMPT.format(message=message))
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    label, confidence, reason = "", 0.0, ""
    matched = re.search(r"\{.*\}", text, re.DOTALL)
    if matched:
        try:
            obj = json.loads(matched.group(0))
            label = str(obj.get("label", "")).strip()
            confidence = float(obj.get("confidence", 0.0) or 0.0)
            reason = str(obj.get("reason", "")).strip()
        except (ValueError, TypeError):
            pass

    # 兼容模型只回裸 label 的情况
    if label not in _VALID_LABELS:
        for candidate in _VALID_LABELS:
            if f'"{candidate}"' in text:
                label = candidate
                break

    if label not in _VALID_LABELS:
        raise ValueError(f"意图路由返回非法结果：{text[:80]}")
    return RouteDecision(label, reason or "LLM 意图分类", confidence, "llm")


# ── ③ 兜底层 ──
def _fallback_route(reason: str) -> RouteDecision:
    """兜底：一律回退 qa（最安全的路径）。"""
    return RouteDecision(RouteLabel.QA.value, reason, 0.0, "fallback")


def route_intent(message: str) -> RouteDecision:
    """路由入口：规则 → LLM → 兜底，永不抛异常。"""
    decision = _rule_route(message)
    if decision is not None:
        logger.info("intent.rule_hit | label=%s | msg=%s", decision.label, message[:30])
        return decision

    try:
        decision = _llm_route(message)
    except Exception as e:
        logger.warning("intent.llm_failed | error=%s", e)
        return _fallback_route("路由失败，回退问答")

    logger.info(
        "intent.llm_done | label=%s | conf=%.2f | msg=%s",
        decision.label, decision.confidence, message[:30],
    )
    return decision
