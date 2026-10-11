"""入口分流 + 问答 Agent 接口。

核心设计：患者通过「我要咨询」入口显式选择服务，后端按 mode 参数直接路由，
不做自动意图分类（医疗场景误判成本高，显式选择更安全）。

鉴权：所有接口需携带 Bearer Token（见 /api/v1/auth/login）。
"""
import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agents.qa_agent.graph import build_qa_graph
from app.agents.qa_agent.nodes import need_human, retrieve, stream_answer, wants_human
from app.agents.qa_agent.prompts import GENERATE_PROMPT
from app.core.auth import get_current_patient
from app.core.history import save_turn
from app.core.memory import build_thread_id
from app.core.models import PatientProfile
from app.core.qa_flow import force_human_interrupt
from app.core.tracking import remove_pending, track_pending
from app.core import human_session
from app.core.logger import get_logger

router = APIRouter(prefix="/api/v1", tags=["入口分流 / 问答"])
logger = get_logger(__name__)


class ChatRequest(BaseModel):
    mode: str = Field("qa", description="入口标识：qa=咨询问答")
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话 ID，缺省自动生成")


class ChatResponse(BaseModel):
    status: str                       # "done" / "interrupted"
    session_id: str
    thread_id: str
    answer: str = ""
    answer_mode: str = ""
    confidence: Optional[float] = None
    sources: list = []
    interrupt: Optional[dict] = None  # 转人工时，携带给医生看的信息


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, patient: PatientProfile = Depends(get_current_patient)):
    """咨询问答入口：转发到问答 Agent 图（RAG + 人工介入）。

    patient_id 取自登录令牌而非请求体，杜绝伪造身份。
    """
    if req.mode != "qa":
        raise HTTPException(status_code=400, detail=f"未知 mode: {req.mode}")

    patient_id = patient.patient_id
    session_id = req.session_id or str(uuid.uuid4())

    # 人工模式闸门：医生接管中 → 咨询消息直达医生，不再走 AI
    if await human_session.get_active_session(patient_id):
        await human_session.add_message(patient_id, "patient", req.question)
        return ChatResponse(
            status="queued_to_doctor",
            session_id=session_id,
            thread_id="",
            answer="已收到您的问题，正在由医生为您解答，请稍候。",
        )

    # 患者主动要求人工：跳过检索，直接转人工
    if wants_human(req.question):
        tid, reason = await force_human_interrupt(
            patient_id, session_id, req.question, "患者主动要求人工接入"
        )
        return ChatResponse(
            status="interrupted", session_id=session_id, thread_id=tid,
            interrupt={"reason": reason},
        )

    thread_id = build_thread_id(patient_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}

    graph = build_qa_graph()
    initial_state = {
        "messages": [HumanMessage(content=req.question)],
        "patient_id": patient_id,
        "session_id": session_id,
        "question": req.question,
        "confidence": 0.0,
    }

    result = await graph.ainvoke(initial_state, config=config)

    # 命中 interrupt()：图暂停，等待医生介入
    interrupt_payload = result.get("__interrupt__")
    if interrupt_payload:
        payload = _unwrap_interrupt(interrupt_payload)
        await track_pending(thread_id, patient_id, session_id, req.question, payload)
        logger.info("chat.interrupted | thread_id=%s", thread_id)
        return ChatResponse(
            status="interrupted",
            session_id=session_id,
            thread_id=thread_id,
            interrupt=payload,
        )

    answer = result.get("answer", "")
    await _persist_turn(patient_id, session_id, req.mode, req.question, answer)

    return ChatResponse(
        status="done",
        session_id=session_id,
        thread_id=thread_id,
        answer=answer,
        answer_mode=result.get("answer_mode", "rag"),
        confidence=result.get("confidence"),
        sources=result.get("sources", []),
    )


def _sse(payload: dict) -> dict:
    """把事件字典封装成 SSE data 行（data 字段存 JSON 字符串）。"""
    return {"data": json.dumps(payload, ensure_ascii=False)}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, patient: PatientProfile = Depends(get_current_patient)):
    """咨询问答入口（SSE 流式）：检索 → 逐 token 推送回答；转人工则返回 interrupted 事件。

    事件类型：
      meta        检索元信息（confidence + sources）
      token       回答文本增量（逐字）
      done        正常结束
      interrupted 已转人工（含 thread_id / reason）
    """
    if req.mode != "qa":
        raise HTTPException(status_code=400, detail=f"未知 mode: {req.mode}")

    patient_id = patient.patient_id
    session_id = req.session_id or str(uuid.uuid4())
    question = req.question

    async def event_source():
        # ⓪ 人工模式闸门：医生接管中 → 咨询消息直达医生，不再走 AI
        if await human_session.get_active_session(patient_id):
            await human_session.add_message(patient_id, "patient", question)
            yield _sse({
                "type": "queued_to_doctor",
                "message": "已发送给医生，正在等待医生回复…",
            })
            yield _sse({"type": "done"})
            return

        # ① 患者主动要求人工：跳过检索，直接转人工
        if wants_human(question):
            tid, reason = await force_human_interrupt(
                patient_id, session_id, question, "患者主动要求人工接入"
            )
            yield _sse({"type": "interrupted", "thread_id": tid, "reason": reason})
            return

        # ② 检索 + 路由判断（先告知前端，避免干等）
        yield _sse({"type": "status", "content": "正在检索知识库…"})
        r = await retrieve(patient_id, question)
        confidence = r["confidence"]
        category = r["query_category"]

        # ② 转人工：走完整 LangGraph 图拿到 interrupt（保留医生 resume 能力）
        # 通用知识为日常交流，不因无命中而转人工
        if category != "通用知识" and need_human(confidence, r["is_high_risk"]):
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
                },
                config=config,
            )
            interrupt_payload = result.get("__interrupt__")
            payload = _unwrap_interrupt(interrupt_payload) if interrupt_payload else {"message": "已转人工"}
            await track_pending(thread_id, patient_id, session_id, question, payload)
            yield _sse({
                "type": "interrupted",
                "thread_id": thread_id,
                "reason": payload.get("reason", ""),
            })
            return

        # ③ 快车道命中：直接输出标准答案（无需大模型生成）
        if r["source"] == "faq":
            yield _sse({
                "type": "meta",
                "source": "faq",
                "confidence": round(confidence, 4),
                "sources": [],
                "query_category": category,
                "strategy": r["strategy"],
            })
            yield _sse({"type": "token", "content": r["faq_answer"]})
            await _persist_turn(patient_id, session_id, req.mode, question, r["faq_answer"])
            yield _sse({"type": "done"})
            return

        # ④ 正常：流式生成（context 为 RAG 侧已融合好的参考资料）
        yield _sse({
            "type": "meta",
            "source": r["source"],
            "confidence": round(confidence, 4),
            "sources": r["sources"],
            "query_category": category,
            "strategy": r["strategy"],
        })

        prompt = GENERATE_PROMPT.format_messages(
            context=r["context"] or "（未检索到相关资料）",
            doctor_note="（无）",
            question=question,
        )

        parts = []
        async for piece in stream_answer(prompt):
            parts.append(piece)
            yield _sse({"type": "token", "content": piece})
            await asyncio.sleep(0.01)

        await _persist_turn(patient_id, session_id, req.mode, question, "".join(parts))
        yield _sse({"type": "done"})

    return EventSourceResponse(event_source())


async def _persist_turn(patient_id: str, session_id: str, mode: str, question: str, answer: str) -> None:
    """会话落库（尽力而为，失败不影响主流程）。"""
    try:
        await save_turn(patient_id, session_id, mode, question, answer)
    except Exception as e:
        logger.warning("chat.history_save_failed | error=%s", e)


def _unwrap_interrupt(interrupt_payload) -> dict:
    """兼容不同 LangGraph 版本对 __interrupt__ 的包装结构，取出 interrupt(value)。"""
    if isinstance(interrupt_payload, (tuple, list)):
        for item in interrupt_payload:
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                return value
            return {"message": str(value)}
    if isinstance(interrupt_payload, dict):
        return interrupt_payload
    return {"message": str(interrupt_payload)}