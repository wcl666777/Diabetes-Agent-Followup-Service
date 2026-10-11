"""人工介入（Human-in-the-Loop）待处理记录：持久化到 MySQL。

LangGraph 的 interrupt() 暂停时，graph 状态由 MemorySaver 内存保存（进程内），
而「医生端待处理列表」需要跨进程可见、服务重启不丢，故落库到 hitl_pending 表。
"""
from typing import Optional

from sqlalchemy import delete, select

from app.core.db import get_session_factory
from app.core.logger import get_logger
from app.core.models import HitlPending

logger = get_logger(__name__)


async def track_pending(
    thread_id: str,
    patient_id: str,
    session_id: str,
    question: str,
    payload: Optional[dict] = None,
) -> None:
    """登记一条待人工介入记录（按 thread_id 幂等，重复登记覆盖）。"""
    factory = get_session_factory()
    async with factory() as session:
        existing = await session.execute(
            select(HitlPending).where(HitlPending.thread_id == thread_id)
        )
        row = existing.scalar_one_or_none()
        if row:
            row.patient_id = patient_id
            row.session_id = session_id
            row.question = question
            row.payload = payload or {}
            row.status = "pending"   # 同一 thread_id 再次转人工：重置为待处理
            row.answer = ""
        else:
            session.add(
                HitlPending(
                    thread_id=thread_id,
                    patient_id=patient_id,
                    session_id=session_id,
                    question=question,
                    payload=payload or {},
                )
            )
        await session.commit()


async def get_pending(thread_id: str) -> Optional[dict]:
    """按 thread_id 查询待处理记录。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HitlPending).where(HitlPending.thread_id == thread_id)
        )
        row = result.scalar_one_or_none()
    return _to_dict(row) if row else None


async def list_pending() -> list[dict]:
    """列出「待处理」记录（医生端列表，仅未完成的）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HitlPending)
            .where(HitlPending.status == "pending")
            .order_by(HitlPending.created_at.desc())
        )
        rows = result.scalars().all()
    return [_to_dict(r) for r in rows]


async def complete_pending(thread_id: str, answer: str) -> None:
    """医生恢复后：标记为已完成并写入最终回答（保留供患者端轮询取回）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HitlPending).where(HitlPending.thread_id == thread_id)
        )
        row = result.scalar_one_or_none()
        if row:
            row.status = "done"
            row.answer = answer
        await session.commit()


async def get_result(thread_id: str) -> Optional[dict]:
    """查询某次人工介入的结果（患者端轮询用）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(HitlPending).where(HitlPending.thread_id == thread_id)
        )
        row = result.scalar_one_or_none()
    return _to_dict(row) if row else None


async def remove_pending(thread_id: str) -> None:
    """删除待处理记录（清理用）。"""
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(delete(HitlPending).where(HitlPending.thread_id == thread_id))
        await session.commit()


def _to_dict(row: HitlPending) -> dict:
    return {
        "thread_id": row.thread_id,
        "patient_id": row.patient_id,
        "session_id": row.session_id,
        "question": row.question,
        "payload": row.payload or {},
        "status": row.status or "pending",
        "answer": row.answer or "",
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }