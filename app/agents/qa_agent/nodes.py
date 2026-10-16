"""问答 Agent 的节点实现。

流程：retrieve（MCP 拉公共指南 + 本地取个人出院小结）→ 路由判断 → generate / human_review(interrupt)。
"""
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from app.agents.qa_agent.prompts import GENERATE_PROMPT
from app.agents.qa_agent.state import QAState
from app.config import get_settings
from app.core import mcp_client
from app.core.llm_factory import get_llm
from app.core.logger import get_logger
from app.core.question_types import classify_question_type

logger = get_logger(__name__)

# 高危关键词：命中即视为高危问题，无论检索置信度如何都转人工
_HIGH_RISK_KEYWORDS = [
    "停药", "加药", "换药", "调整用药", "胰岛素", "昏迷", "酮症",
    "呼吸困难", "抽搐", "严重低血糖", "急救", "怀孕", "手术",
]


def _is_high_risk(question: str) -> bool:
    return any(kw in question for kw in _HIGH_RISK_KEYWORDS)


# 患者主动要求人工的关键词：命中即跳过检索，直接转人工
_HUMAN_REQUEST_KEYWORDS = [
    "找医生", "转人工", "要医生", "请医生", "人工客服", "人工服务", "真人",
    "联系医生", "转接医生", "问医生", "人工回答", "叫医生", "医生回复",
]


def wants_human(question: str) -> bool:
    """患者是否明确要求人工接入。"""
    return any(kw in question for kw in _HUMAN_REQUEST_KEYWORDS)


# 细粒度问题类型（饮食 / 运动 / 用药…）：由「意图识别 Agent」在主流程前识别，
# 结果通过 state["question_type"] 传入。此处仅保留规则兜底（意图 Agent 未跑或降级时使用）。
# 与 RAG 侧「通用知识 / 专业咨询」二分类互补：RAG 决定要不要检索、走哪种检索策略，
# 问题类型决定生成时的答题侧重，二者互不重叠。


async def _get_discharge_summary(patient_id: str) -> str:
    """从 PostgreSQL 获取患者出院小结（个人源）；DB 不可用则返回空。"""
    try:
        from app.core.db import get_session_factory
        from app.core.models import PatientProfile
        from sqlalchemy import select

        async with get_session_factory()() as session:
            result = await session.execute(
                select(PatientProfile).where(PatientProfile.patient_id == patient_id)
            )
            row = result.scalar_one_or_none()
            return row.discharge_summary if row else ""
    except Exception as e:
        logger.warning("qa.get_discharge_summary_failed | error=%s", e)
        return ""


async def retrieve(patient_id: str, question: str) -> dict:
    """检索：①本地取个人资料（病例 + 出院小结）②通过 MCP 走 RAG 主流程。

    RAG 主流程 = 高频快车道（Redis 缓存 + MySQL BM25）→ 意图识别 → 检索策略 →
    混合检索 → 融合个人资料。个人资料由本函数本地检索后传给 RAG 服务统一融合，
    生成只在 Agent 侧发生一次。

    抽成独立函数，供 LangGraph 节点与 SSE 流式接口复用，避免两处重复实现。
    """
    # ① 个人文档（病例 + 出院小结，本地 MySQL，父子块检索：child 命中 → parent 上下文）
    try:
        from app.core.patient_docs import retrieve_personal

        personal_chunks = await retrieve_personal(patient_id, question, top_k=3)
        personal = "\n\n".join(
            f"[{c['doc_type_label']}/{c['title']}] {c['content']}" for c in personal_chunks
        )
    except Exception as e:
        logger.warning("qa.retrieve_personal_failed | error=%s", e)
        personal = ""

    # 未上传个人文档时，回退到档案里的出院小结
    if not personal:
        personal = await _get_discharge_summary(patient_id)

    # ② 公共知识 + 快车道 + 意图识别 + 检索策略（走 MCP 备料工具，个人资料一并传入融合）
    result: dict = {}
    try:
        result = await mcp_client.rag_query(question, personal_context=personal)
    except Exception as e:
        logger.warning("qa.retrieve_mcp_failed | error=%s", e)

    confidence = float(result.get("confidence", 0.0))
    logger.info(
        "qa.retrieve_done | question=%s | source=%s | cat=%s | strategy=%s | conf=%.3f | high_risk=%s",
        question[:30], result.get("source", "none"), result.get("query_category", ""),
        result.get("strategy", ""), confidence, _is_high_risk(question),
    )

    return {
        "confidence": confidence,
        "personal_context": personal,
        "is_high_risk": _is_high_risk(question),
        "query_category": result.get("query_category", "专业咨询"),
        "strategy": result.get("strategy", ""),
        "context": result.get("context", ""),
        "source": result.get("source", "none"),
        "faq_answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }


async def retrieve_node(state: QAState) -> dict:
    """检索节点：复用 retrieve()，仅是 LangGraph 节点的适配壳。"""
    if state.get("force_human"):
        # 患者主动要求人工：跳过检索，直接置为高危以触发 human_review
        return {
            "confidence": 0.0, "personal_context": "", "is_high_risk": True,
            "query_category": "专业咨询", "strategy": "", "context": "",
            "source": "none", "faq_answer": "", "sources": [],
        }
    if state.get("retrieval_precomputed"):
        # 上游接口已检索并注入结果（用于转人工前的判定），直接复用，避免二次检索
        return {}
    return await retrieve(state["patient_id"], state["question"])


def need_human(confidence: float, is_high_risk: bool) -> bool:
    """判断是否需要转人工：高危或检索置信度低于阈值。"""
    return is_high_risk or confidence < get_settings().qa_confidence_threshold


async def stream_answer(prompt) -> AsyncIterator[str]:
    """流式生成：逐 token 产出文本增量（供 SSE 推送）。

    使用 streaming 模式调用 DeepSeek，逐 chunk 返回。
    """
    llm = get_llm("qa", temperature=0.2, streaming=True)
    async for chunk in llm.astream(prompt):
        content = chunk.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list) and content:
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = ""
        if text:
            yield text


def _route_after_retrieve(state: QAState) -> str:
    """检索后路由：主动要求人工 / 高危 / 低置信 → 人工介入；通用知识、快车道命中 → 直接生成。"""
    if state.get("force_human"):
        return "human_review"
    # 通用知识（日常交流）无需资料，直接由大模型回答，不因无命中而转人工
    if state.get("query_category") == "通用知识":
        return "generate"
    threshold = get_settings().qa_confidence_threshold
    if state.get("is_high_risk") or state.get("confidence", 0.0) < threshold:
        return "human_review"
    return "generate"


def human_review_node(state: QAState) -> dict:
    """人工介入节点：调用 interrupt() 暂停图，等待医生输入后恢复。

    interrupt() 返回医生通过 Command(resume=...) 传入的内容。
    这是真正的 Human-in-the-Loop：状态由 checkpointer 持久化，恢复不重跑前置节点。
    """
    reason = (
        "患者主动要求人工接入" if state.get("force_human")
        else ("高危问题" if state.get("is_high_risk") else "检索置信度不足")
    )
    decision = interrupt(
        {
            "type": "human_review_request",
            "question": state["question"],
            "retrieved_context": state.get("context", "") or "（无检索资料）",
            "sources": state.get("sources", []),
            "confidence": state.get("confidence", 0.0),
            "reason": reason,
        }
    )
    doctor_note = decision.get("doctor_note", "") if isinstance(decision, dict) else str(decision)
    logger.info("qa.human_review_resumed | note=%s", doctor_note[:30])
    # 医生接管：直接以医生输入作为最终回答，不再调用大模型生成
    answer = (doctor_note or "").strip() or "医生已介入本次咨询，请遵医嘱，如有不适请及时就医。"
    return {
        "need_human": True,
        "doctor_note": doctor_note,
        "answer": answer,
        "answer_mode": "human",
    }


async def generate_node(state: QAState) -> dict:
    """生成节点：用 RAG 侧融合好的上下文生成回答（通用知识时为日常交流提示）。"""
    # 快车道命中：标准答案直接返回，不再调用大模型
    if state.get("source") == "faq" and state.get("faq_answer"):
        answer = state["faq_answer"]
        return {"answer": answer, "sources": [], "messages": [AIMessage(content=answer)]}

    # 问题类型优先取「意图识别 Agent」的识别结果；缺失时退回规则分类
    question_type = state.get("question_type") or classify_question_type(state["question"])
    prompt = GENERATE_PROMPT.format_messages(
        question_type=question_type,
        context=state.get("context") or "（未检索到相关资料）",
        doctor_note=state.get("doctor_note") or "（无）",
        question=state["question"],
    )

    llm = get_llm("qa", temperature=0.2)
    response = await llm.ainvoke(prompt)
    answer = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "answer": answer,
        "sources": state.get("sources", []),
        "question_type": question_type,
        "messages": [AIMessage(content=answer)],
    }