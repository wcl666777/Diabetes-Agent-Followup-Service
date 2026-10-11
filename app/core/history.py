"""会话历史持久化：写入 / 读取 MySQL 的 chat_history 表。"""
from typing import Optional

from sqlalchemy import select

from app.core.db import get_session_factory
from app.core.logger import get_logger
from app.core.models import ChatHistory

logger = get_logger(__name__)


async def save_turn(
    patient_id: str,
    session_id: str,
    mode: str,
    question: str,
    answer: str,
) -> None:
    """写入一轮对话（一问一答）。"""
    factory = get_session_factory()
    async with factory() as session:
        session.add(
            ChatHistory(
                patient_id=patient_id,
                session_id=session_id,
                mode=mode,
                question=question,
                answer=answer,
            )
        )
        await session.commit()


async def load_history(patient_id: str, session_id: str, limit: int = 10) -> list[dict]:
    """读取最近 N 轮历史，供多轮问答回放。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(ChatHistory)
            .where(ChatHistory.patient_id == patient_id, ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    # 按时间升序返回（最早在前）
    return [
        {"question": r.question, "answer": r.answer}
        for r in reversed(rows)
    ]