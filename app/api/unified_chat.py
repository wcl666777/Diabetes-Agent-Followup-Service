"""统一聊天入口。

对用户一句话，自动判断该交给哪个 Agent，并以 SSE 流式返回。流程：
  ① 规则前置拦截：问候 / 感谢 / 道别 / 身份 / 功能咨询 —— 零 Token 直接模板回复；
  ② 意图路由（三层：规则 → LLM 结构化分类 → 兜底，见 core/intent_router.py）：返回 label ∈ {qa, glucose, clarify}；
  ③ 分发：
       qa       → 编排两步：先跑「意图识别 Agent」（识别饮食 / 运动 / 用药…），
                  再把识别结果喂给问答 Agent（检索 → 流式生成 / 必要时转人工）
       glucose  → 返回引导卡片，提示到「记录血糖」页签填写
       clarify  → 追问澄清

SSE 事件：routing_decision / status / token / meta / guidance / done / error
"""
import asyncio
import json
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agents.qa_agent.graph import build_qa_graph
from app.agents.qa_agent.nodes import (
    need_human,
    retrieve,
    stream_answer,
    wants_human,
)
from app.agents.qa_agent.prompts import GENERATE_PROMPT
from app.core.auth import get_current_patient
from app.core.history import save_turn
from app.core.intent_router import route_intent
from app.core.logger import get_logger
from app.core.memory import build_thread_id
from app.core.models import PatientProfile
from app.core.orchestrator import get_orchestrator
from app.core.qa_flow import force_human_interrupt
from app.core.question_types import classify_question_type
from app.core.tracking import get_result, track_pending
from app.core import human_session

router = APIRouter(prefix="/api/v1/chat", tags=["统一聊天入口"])
logger = get_logger(__name__)


class UnifiedChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="")

    def resolved_session(self) -> str:
        return self.session_id or str(uuid.uuid4())


# ── ① 规则前置拦截（零 Token，覆盖高频寒暄）──
_RULES = [
    (re.compile(r"^(你好|您好|hi|hello|嗨|在吗|哈喽)"), "您好，我是您的糖尿病随访助手，可以咨询饮食、用药、血糖相关问题。请问有什么可以帮您？"),
    (re.compile(r"(谢谢|感谢|多谢|thank)"), "不客气，祝您血糖平稳、身体健康！还有其他问题随时问我。"),
    (re.compile(r"(再见|拜拜|bye|退出)"), "好的，再见！请记得按时监测血糖、规律用药。"),
    (re.compile(r"(你是谁|什么助手|介绍一下你)"), "我是糖尿病出院患者智能随访助手，由「问答 Agent」和「血糖管理 Agent」协作，可解答医疗咨询并记录分析血糖。"),
    (re.compile(r"(你能做什么|有什么功能|帮助)"), "我可以：1) 解答糖尿病相关的医疗咨询（基于权威指南）；2) 记录并解读您的血糖；3) 复杂问题转人工由医生跟进。"),
]


def _pre_filter(text: str) -> str:
    """规则命中则返回模板回复，否则返回空串。"""
    t = text.strip().lower()
    for pattern, reply in _RULES:
        if pattern.search(t):
            return reply
    return ""


# ── ② 意图路由（三层：规则 → LLM 结构化分类 → 兜底，见 core/intent_router.py）──
_LABEL_DISPLAY = {"qa": "问答助手", "glucose": "血糖管理", "clarify": "意图澄清"}


def _sse(payload: dict) -> dict:
    return {"data": json.dumps(payload, ensure_ascii=False)}


async def _stream_qa(patient_id: str, session_id: str, question: str):
    """问答分支的流式生成（与 /chat/stream 同源逻辑）。

    编排：先跑「意图识别 Agent」拿到问题类型，再交给问答 Agent 生成（两步串联）。
    """
    # ⓪ 编排第一步：意图识别 Agent（识别饮食 / 运动 / 用药…），结果喂给问答 Agent
    yield _sse({"type": "status", "content": "正在理解问题…"})
    intent = await get_orchestrator().run_intent(patient_id, session_id, question)
    question_type = intent.get("question_type") or classify_question_type(question)

    yield _sse({"type": "status", "content": "正在检索知识库…"})
    r = await retrieve(patient_id, question)
    confidence = r["confidence"]
    category = r["query_category"]

    # 转人工（通用知识为日常交流，不因无命中而转人工）
    if category != "通用知识" and need_human(confidence, r["is_high_risk"]):
        thread_id = build_thread_id(patient_id, session_id)
        config = {"configurable": {"thread_id": thread_id}}
        graph = build_qa_graph()
        # 复用上面已完成的检索结果，避免节点内二次检索，并保证路由与本次判定一致
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "patient_id": patient_id,
            "session_id": session_id,
            "question": question,
            "retrieval_precomputed": True,
            "question_type": question_type,
            **r,
        }
        result = await graph.ainvoke(initial_state, config=config)
        payload = result.get("__interrupt__")
        reason = "高危问题或检索置信度不足"
        if payload:
            item = payload[0] if isinstance(payload, (tuple, list)) else payload
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                reason = value.get("reason", reason)
        await track_pending(thread_id, patient_id, session_id, question, {"reason": reason})
        yield _sse({"type": "interrupted", "thread_id": thread_id, "reason": reason})
        return

    # 快车道命中：直接输出标准答案（无需大模型生成）
    if r["source"] == "faq":
        yield _sse({
            "type": "meta",
            "source": "faq",
            "confidence": round(confidence, 4),
            "sources": [],
            "query_category": category,
            "strategy": r["strategy"],
            "question_type": question_type,
        })
        yield _sse({"type": "token", "content": r["faq_answer"]})
        try:
            await save_turn(patient_id, session_id, "qa", question, r["faq_answer"])
        except Exception as e:
            logger.warning("unified.save_failed | error=%s", e)
        yield _sse({"type": "done"})
        return

    # RAG 生成：context 为 RAG 侧已融合好的参考资料
    yield _sse({
        "type": "meta",
        "source": r["source"],
        "confidence": round(confidence, 4),
        "sources": r["sources"],
        "query_category": category,
        "strategy": r["strategy"],
        "question_type": question_type,
    })

    prompt = GENERATE_PROMPT.format_messages(
        question_type=question_type,
        context=r["context"] or "（未检索到相关资料）",
        doctor_note="（无）",
        question=question,
    )
    parts = []
    async for piece in stream_answer(prompt):
        parts.append(piece)
        yield _sse({"type": "token", "content": piece})
        await asyncio.sleep(0.01)

    try:
        await save_turn(patient_id, session_id, "qa", question, "".join(parts))
    except Exception as e:
        logger.warning("unified.save_failed | error=%s", e)
    yield _sse({"type": "done"})


@router.post("/unified/stream")
async def unified_stream(req: UnifiedChatRequest, patient: PatientProfile = Depends(get_current_patient)):
    """统一入口：自动路由到合适的 Agent 并流式返回。"""
    patient_id = patient.patient_id
    session_id = req.resolved_session()
    message = req.message

    async def event_source():
        # ⓪ 患者主动要求人工：跳过路由与检索，直接转人工
        if wants_human(message):
            yield _sse({"type": "routing_decision", "label": "human", "agent": "人工接入", "reason": "患者主动要求"})
            tid, reason = await force_human_interrupt(
                patient_id, session_id, message, "患者主动要求人工接入"
            )
            yield _sse({"type": "interrupted", "thread_id": tid, "reason": reason})
            return

        # ① 规则前置
        preset = _pre_filter(message)
        if preset:
            yield _sse({"type": "routing_decision", "label": "preset", "agent": "规则回复", "reason": "命中寒暄规则"})
            yield _sse({"type": "token", "content": preset})
            yield _sse({"type": "done", "session_id": session_id})
            return

        # ② 意图路由（三层，放线程池避免阻塞事件循环）
        loop = asyncio.get_running_loop()
        decision = await loop.run_in_executor(None, route_intent, message)
        label = decision.label
        yield _sse({
            "type": "routing_decision",
            "label": label,
            "agent": _LABEL_DISPLAY.get(label, label),
            "reason": decision.reason,
            "confidence": decision.confidence,
            "source": decision.source,
        })

        # ③ 分发
        if label == "qa":
            # 人工模式闸门：医生接管中 → 咨询消息直达医生，不再走 AI（血糖记录不受影响）
            if await human_session.get_active_session(patient_id):
                await human_session.add_message(patient_id, "patient", message)
                yield _sse({
                    "type": "queued_to_doctor",
                    "message": "已发送给医生，正在等待医生回复…",
                })
                yield _sse({"type": "done", "session_id": session_id})
                return
            async for ev in _stream_qa(patient_id, session_id, message):
                yield ev
        elif label == "glucose":
            yield _sse({
                "type": "guidance",
                "content": "检测到您要记录血糖。请切换到「记录血糖」页签，填入血糖值与测量类型后提交，我会立即为您解读并告警。",
            })
            yield _sse({"type": "done", "session_id": session_id})
        else:
            yield _sse({
                "type": "token",
                "content": "抱歉，我没太理解您的意思。请问您是想咨询健康问题，还是要记录血糖呢？",
            })
            yield _sse({"type": "done", "session_id": session_id})

    return EventSourceResponse(event_source())


@router.get("/hitl/{thread_id}")
async def hitl_result(thread_id: str, patient: PatientProfile = Depends(get_current_patient)):
    """患者端轮询：查询某次转人工是否已被医生处理，返回最终回答（实现闭环）。"""
    rec = await get_result(thread_id)
    if not rec:
        raise HTTPException(status_code=404, detail="无此人工介入记录")
    if rec["patient_id"] != patient.patient_id:
        raise HTTPException(status_code=403, detail="无权访问该记录")
    return {"status": rec["status"], "answer": rec["answer"]}
