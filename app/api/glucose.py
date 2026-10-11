"""血糖管理 Agent 接口。"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agents.glucose_agent.graph import build_glucose_graph
from app.core.auth import get_current_patient
from app.core.history import save_turn
from app.core.memory import build_thread_id
from app.core.models import PatientProfile
from app.core.logger import get_logger

router = APIRouter(prefix="/api/v1", tags=["入口分流 / 血糖"])
logger = get_logger(__name__)


class GlucoseRequest(BaseModel):
    mode: str = Field("glucose", description="入口标识：glucose=记录血糖")
    value: float = Field(..., gt=0, le=60, description="血糖值 mmol/L")
    measurement_type: str = Field("空腹", description="空腹/餐后/睡前/随机")
    session_id: Optional[str] = Field(None, description="会话 ID，缺省自动生成")


class GlucoseResponse(BaseModel):
    status: str
    session_id: str
    alert_level: str
    interpretation: str
    advice: str
    metrics: dict
    saved: bool


@router.post("/glucose", response_model=GlucoseResponse)
async def glucose(req: GlucoseRequest, patient: PatientProfile = Depends(get_current_patient)):
    """记录血糖入口：转发到血糖管理 Agent 图（确定性计算 + 告警）。"""
    if req.mode != "glucose":
        raise HTTPException(status_code=400, detail=f"未知 mode: {req.mode}")

    patient_id = patient.patient_id
    session_id = req.session_id or str(uuid.uuid4())
    thread_id = build_thread_id(patient_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}

    graph = build_glucose_graph()
    initial_state = {
        "messages": [],
        "patient_id": patient_id,
        "session_id": session_id,
        "value_mmol": req.value,
        "measurement_type": req.measurement_type,
    }

    result = await graph.ainvoke(initial_state, config=config)

    interpretation = result.get("interpretation", "")
    advice = result.get("advice", "")
    question = f"血糖 {req.value} mmol/L（{req.measurement_type}）"

    # 会话落库（尽力而为）
    try:
        await save_turn(patient_id, session_id, req.mode, question, f"{interpretation}\n{advice}")
    except Exception as e:
        logger.warning("glucose.history_save_failed | error=%s", e)

    return GlucoseResponse(
        status="done",
        session_id=session_id,
        alert_level=result.get("alert_level", "normal"),
        interpretation=interpretation,
        advice=advice,
        metrics=result.get("metrics", {}),
        saved=result.get("saved", False),
    )