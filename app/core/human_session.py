"""人工接管会话状态机（Human-in-the-Loop 的「接管式连续会话」）。

与 tracking.py 的分工：
  - tracking.py：单次转人工的待处理记录（hitl_pending），负责「医生第一次回复」。
  - 本模块：医生接管后的「人工模式」状态（human_sessions）+ 双方多轮对话（human_messages）。

状态流转：
  AI 问答 ──低置信/高危──► 待处理(hitl_pending) ──医生首次回复──► 人工模式(active)
         ◄────────── 医生「结束诊断」(closed) ──────────┘

锁定粒度：按 patient_id。同一患者进入人工模式后，其咨询消息直达医生，
直到医生结束；再次开启时清空上一次的对话，保证一次接管一段连续会话。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select

from app.core.db import get_session_factory
from app.core.logger import get_logger
from app.core.models import HumanMessage, HumanSession

logger = get_logger(__name__)


# ── 会话（状态机）──
async def get_active_session(patient_id: str) -> Optional[dict]:
    """查询该患者当前是否处于人工模式（active）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HumanSession)
            .where(HumanSession.patient_id == patient_id, HumanSession.status == "active")
            .order_by(HumanSession.id.desc())
        )
        row = result.scalars().first()
    return _session_to_dict(row) if row else None


async def get_latest_session(patient_id: str) -> Optional[dict]:
    """查询该患者最近一次人工会话（含已结束的），供患者端判断状态切换。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HumanSession)
            .where(HumanSession.patient_id == patient_id)
            .order_by(HumanSession.id.desc())
        )
        row = result.scalars().first()
    return _session_to_dict(row) if row else None


async def open_session(patient_id: str, doctor_id: str, thread_id: str = "") -> dict:
    """医生首次回复即接管：开启（或复用）人工模式，并清空上一轮对话。"""
    existing = await get_active_session(patient_id)
    if existing:
        return existing

    factory = get_session_factory()
    async with factory() as session:
        # 清空该患者上一轮人工对话，保证「一次接管 = 一段连续会话」
        await session.execute(delete(HumanMessage).where(HumanMessage.patient_id == patient_id))
        row = HumanSession(
            patient_id=patient_id, doctor_id=doctor_id, thread_id=thread_id, status="active"
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
    logger.info("human_session.opened | patient=%s | doctor=%s", patient_id, doctor_id)
    return _session_to_dict(row)


async def close_session(patient_id: str, doctor_id: str = "") -> bool:
    """医生「结束诊断」：关闭人工模式，患者恢复 AI 问答。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HumanSession)
            .where(HumanSession.patient_id == patient_id, HumanSession.status == "active")
        )
        rows = result.scalars().all()
        for row in rows:
            row.status = "closed"
            row.closed_at = datetime.utcnow()
            if doctor_id:
                row.doctor_id = doctor_id
        await session.commit()
    if rows:
        logger.info("human_session.closed | patient=%s", patient_id)
    return bool(rows)


# ── 对话消息 ──
async def add_message(patient_id: str, role: str, content: str) -> dict:
    """追加一条人工对话消息（role=patient / doctor）。"""
    factory = get_session_factory()
    async with factory() as session:
        row = HumanMessage(patient_id=patient_id, role=role, content=content)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return _message_to_dict(row)


async def list_messages(patient_id: str, limit: int = 200) -> list[dict]:
    """按时间正序返回该患者当前人工会话的全部消息。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HumanMessage)
            .where(HumanMessage.patient_id == patient_id)
            .order_by(HumanMessage.id.asc())
            .limit(limit)
        )
        rows = result.scalars().all()
    return [_message_to_dict(r) for r in rows]


async def get_patient_view(patient_id: str) -> dict:
    """患者端视图：当前是否处于人工模式 + 对话记录。"""
    latest = await get_latest_session(patient_id)
    active = bool(latest and latest["status"] == "active")
    messages = await list_messages(patient_id) if latest else []
    return {
        "active": active,
        "status": latest["status"] if latest else "none",
        "doctor_id": latest["doctor_id"] if latest else "",
        "messages": messages,
    }


async def list_active_sessions() -> list[dict]:
    """医生端视图：所有进行中的人工会话（含对话记录）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HumanSession)
            .where(HumanSession.status == "active")
            .order_by(HumanSession.id.desc())
        )
        rows = result.scalars().all()

    sessions = []
    for row in rows:
        d = _session_to_dict(row)
        d["messages"] = await list_messages(row.patient_id)
        sessions.append(d)
    return sessions


def _session_to_dict(row: HumanSession) -> dict:
    return {
        "id": row.id,
        "patient_id": row.patient_id,
        "doctor_id": row.doctor_id or "",
        "thread_id": row.thread_id or "",
        "status": row.status or "active",
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "closed_at": row.closed_at.isoformat() if row.closed_at else None,
    }


def _message_to_dict(row: HumanMessage) -> dict:
    return {
        "id": row.id,
        "role": row.role or "patient",
        "content": row.content or "",
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
