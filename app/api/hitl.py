"""人工介入（Human-in-the-Loop）接口。

两条链路：
  ① 单次转人工：问答 Agent 在低置信/高危时 interrupt() 暂停 → 医生 /pending 看到 →
     /resume 用 Command(resume=...) 恢复，图不重跑前置节点。
  ② 接管式连续会话：医生 /resume（首次回复）后自动开启「人工模式」——
     之后患者的咨询消息直达医生（见 chat/unified_chat 的闸门），
     双方通过 /session/reply 多轮交流，医生 /session/close 结束诊断后恢复 AI。

鉴权：pending / resume / sessions / session/reply / session/close 仅 doctor；
      session/me（轮询自己的会话）仅 patient。
"""
from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.qa_agent.graph import build_qa_graph
from app.core.auth import get_current_doctor, get_current_patient
from app.core import human_session, tracking
from app.core.logger import get_logger
from app.core.models import PatientProfile

router = APIRouter(prefix="/api/v1/hitl", tags=["人工介入"])
logger = get_logger(__name__)


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="中断时返回的 thread_id")
    action: str = Field("override", description="医生动作：override=人工接管")
    doctor_note: str = Field("", description="医生回复内容（将直接作为最终回答，不再经大模型生成）")


class ResumeResponse(BaseModel):
    status: str
    answer: str
    answer_mode: str
    doctor_note: str


class SessionReplyRequest(BaseModel):
    patient_id: str = Field(..., description="接管会话所属患者")
    content: str = Field(..., min_length=1, max_length=2000, description="医生回复内容")


class SessionCloseRequest(BaseModel):
    patient_id: str = Field(..., description="要结束诊断的患者")


@router.get("/pending")
async def pending(doctor: PatientProfile = Depends(get_current_doctor)):
    """医生端：列出所有待人工介入的问答（尚未接管的一次性中断）。"""
    return {"pending": await tracking.list_pending()}


@router.post("/resume", response_model=ResumeResponse)
async def resume(req: ResumeRequest, doctor: PatientProfile = Depends(get_current_doctor)):
    """医生首次回复：恢复被 interrupt() 暂停的图，并开启「人工模式」（接管）。"""
    graph = build_qa_graph()
    config = {"configurable": {"thread_id": req.thread_id}}

    decision = {"action": req.action, "doctor_note": req.doctor_note}

    try:
        result = await graph.ainvoke(Command(resume=decision), config=config)
    except Exception as e:
        logger.error("hitl.resume_failed | error=%s", e)
        raise HTTPException(status_code=500, detail=f"恢复执行失败：{e}")

    answer = result.get("answer", "")
    await tracking.complete_pending(req.thread_id, answer)

    # 首次回复即接管：开启人工模式，并把「原始问题 + 医生回答」作为对话首轮写入
    rec = await tracking.get_pending(req.thread_id)
    patient_id = rec["patient_id"] if rec else ""
    if patient_id:
        await human_session.open_session(patient_id, doctor.patient_id, req.thread_id)
        if rec.get("question"):
            await human_session.add_message(patient_id, "patient", rec["question"])
        if answer:
            await human_session.add_message(patient_id, "doctor", answer)

    logger.info("hitl.resumed | thread_id=%s | patient=%s", req.thread_id, patient_id)

    return ResumeResponse(
        status="done",
        answer=answer,
        answer_mode=result.get("answer_mode", "human"),
        doctor_note=req.doctor_note,
    )


# ── 接管式连续会话 ──
@router.get("/sessions")
async def sessions(doctor: PatientProfile = Depends(get_current_doctor)):
    """医生端：列出所有进行中的人工会话（含完整对话记录）。"""
    return {"sessions": await human_session.list_active_sessions()}


@router.post("/session/reply")
async def session_reply(req: SessionReplyRequest, doctor: PatientProfile = Depends(get_current_doctor)):
    """医生在人工模式下回复患者（可多轮，直到结束诊断）。"""
    active = await human_session.get_active_session(req.patient_id)
    if not active:
        raise HTTPException(status_code=404, detail="该患者当前不在人工模式（可能已结束）")
    msg = await human_session.add_message(req.patient_id, "doctor", req.content)
    return {"status": "ok", "message": msg}


@router.post("/session/close")
async def session_close(req: SessionCloseRequest, doctor: PatientProfile = Depends(get_current_doctor)):
    """医生结束诊断：关闭人工模式，患者恢复 AI 问答。"""
    ok = await human_session.close_session(req.patient_id, doctor.patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="该患者当前无进行中的人工会话")
    return {"status": "closed", "patient_id": req.patient_id}


@router.get("/session/me")
async def session_me(patient: PatientProfile = Depends(get_current_patient)):
    """患者端轮询：是否处于人工模式 + 对话记录（用于接收医生回复 / 感知结束）。"""
    return await human_session.get_patient_view(patient.patient_id)
