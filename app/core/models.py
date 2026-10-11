"""ORM 模型：患者档案 + 血糖记录 + 会话历史 + 人工介入。

全部走 SQLAlchemy 2.0 声明式风格，兼容 MySQL（用 aiomysql 异步驱动）。
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class PatientProfile(Base):
    """患者档案：账号 + 出院小结等个人医疗信息（用于问答 Agent 的「个人源」检索）。"""

    __tablename__ = "patient_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), default="")  # bcrypt 哈希
    role: Mapped[str] = mapped_column(String(16), default="patient")     # patient / doctor
    name: Mapped[str] = mapped_column(String(64), default="")
    discharge_summary: Mapped[str] = mapped_column(Text, default="")  # 出院小结
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GlucoseRecord(Base):
    """血糖记录：原始值 + 确定性计算出的指标 + 告警级别。"""

    __tablename__ = "glucose_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    value_mmol: Mapped[float] = mapped_column(Float)              # 血糖值 mmol/L
    measurement_type: Mapped[str] = mapped_column(String(16), default="空腹")  # 空腹/餐后/睡前
    eag_mgdl: Mapped[float] = mapped_column(Float)                # 换算后的平均血糖 mg/dL
    hba1c_est: Mapped[float] = mapped_column(Float)               # 反推的 HbA1c %
    alert_level: Mapped[str] = mapped_column(String(16), default="normal")  # normal/high/low
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatHistory(Base):
    """多轮会话历史落库：服务重启后会话仍可回放。"""

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    mode: Mapped[str] = mapped_column(String(16), default="qa")   # qa / glucose
    question: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class HitlPending(Base):
    """待人工介入记录（医生端待处理列表，取代内存 dict）。"""

    __tablename__ = "hitl_pending"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), default="")
    question: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)   # 转人工时带给医生的信息
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending / done
    answer: Mapped[str] = mapped_column(Text, default="")              # 医生恢复后的最终回答
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class HumanSession(Base):
    """人工接管会话（会话级状态机）：患者进入人工模式后，其咨询消息直达医生，
    直到医生主动「结束诊断」再恢复 AI 问答。

    - status=active：医生接管中，患者咨询消息不再走 AI
    - status=closed：医生已结束，患者恢复 AI 问答
    """

    __tablename__ = "human_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    doctor_id: Mapped[str] = mapped_column(String(64), default="")
    thread_id: Mapped[str] = mapped_column(String(128), default="")   # 触发转人工的 LangGraph 线程
    status: Mapped[str] = mapped_column(String(16), default="active")  # active / closed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class HumanMessage(Base):
    """人工模式下的对话消息（患者 ↔ 医生，多轮），供双方轮询与重启后回放。"""

    __tablename__ = "human_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16), default="patient")  # patient / doctor
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PatientDocument(Base):
    """患者个人文档：病例 / 出院小结按类型分开存储，仅保留解析后的纯文本全文。

    - doc_type 区分文档类型：discharge_summary（出院小结）/ case_record（病例）
    - content 为解析后的纯文本（不保留原文件），供分块与个人源检索使用
    """

    __tablename__ = "patient_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    doc_type: Mapped[str] = mapped_column(String(32), index=True)  # discharge_summary / case_record
    title: Mapped[str] = mapped_column(String(255), default="")
    content: Mapped[str] = mapped_column(Text, default="")         # 解析后的纯文本全文
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PatientDocChunk(Base):
    """患者文档父子分块：child 块用于检索命中，parent 块用于向大模型提供完整上下文。

    - level=parent：较大的上下文块（如 ~1000 字符），不直接参与检索命中
    - level=child：较小的检索块（如 ~300 字符），向量/关键词检索命中它
    - parent_index：child 指向所属 parent 在本文档内的序号，命中 child 后可回溯 parent
    """

    __tablename__ = "patient_doc_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("patient_documents.id", ondelete="CASCADE"), index=True
    )
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    doc_type: Mapped[str] = mapped_column(String(32), index=True)
    parent_index: Mapped[int] = mapped_column(Integer, default=0)   # 所属 parent 块序号
    level: Mapped[str] = mapped_column(String(16), default="child")  # parent / child
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)