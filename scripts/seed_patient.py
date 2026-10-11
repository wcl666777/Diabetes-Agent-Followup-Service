"""初始化演示数据：写入 3 名患者 + 1 名医生账号、多天血糖记录、多轮会话历史。

用法（在 agent_service 目录执行，需 MySQL 已就绪）：
    python scripts/seed_patient.py

注意：
  - 医生账号用于人工介入接口（/api/v1/hitl/*），需 role=doctor
  - 密码均为演示用途，生产环境务必更换
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 保证以 `python scripts/seed_patient.py` 方式运行时能导入 app 包（项目根目录入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.core.auth import hash_password
from app.core.db import get_session_factory, init_db
from app.core.models import ChatHistory, GlucoseRecord, PatientProfile
from app.core.logger import configure_logging, get_logger
from app.agents.glucose_agent.calculator import (
    classify_alert,
    hba1c_from_eag,
    mmol_to_mgdl,
)

logger = get_logger("seed")

# 演示患者：不同年龄/病程/用药，出院小结各异
PATIENTS = [
    {
        "patient_id": "patient_001",
        "password": "123456",
        "role": "patient",
        "name": "张三",
        "discharge_summary": (
            "患者张三，男，58 岁，2 型糖尿病 5 年。出院诊断：2 型糖尿病、高血压。"
            "用药：二甲双胍 0.5g 每日两次（随餐）、达格列净 10mg 每日一次。"
            "出院医嘱：低盐低脂糖尿病饮食，餐后 1 小时快走 30 分钟；"
            "空腹血糖控制在 4.4~7.0 mmol/L，糖化血红蛋白目标 <7.0%；"
            "注意监测低血糖反应，避免自行停药。"
        ),
    },
    {
        "patient_id": "patient_002",
        "password": "123456",
        "role": "patient",
        "name": "李四",
        "discharge_summary": (
            "患者李四，女，45 岁，2 型糖尿病 2 年，妊娠期糖尿病史。"
            "用药：二甲双胍 0.5g 每日一次（晚餐后）。"
            "出院医嘱：控制总热量，规律三餐，每周 150 分钟中等强度运动；"
            "餐后 2 小时血糖目标 <10.0 mmol/L；备孕需提前复诊调整方案。"
        ),
    },
    {
        "patient_id": "patient_003",
        "password": "123456",
        "role": "patient",
        "name": "王五",
        "discharge_summary": (
            "患者王五，男，67 岁，2 型糖尿病 10 年，糖尿病周围神经病变。"
            "用药：甘精胰岛素 16U 每晚睡前皮下注射，二甲双胍 0.85g 每日一次。"
            "出院医嘱：睡前血糖目标 6.0~8.0 mmol/L，警惕低血糖；"
            "定期足部检查，穿合适鞋袜，避免烫伤。"
        ),
    },
]

DOCTOR = {
    "patient_id": "doctor_001",
    "password": "123456",
    "role": "doctor",
    "name": "王医生",
    "discharge_summary": "内分泌科主治医师（演示账号）。",
}

# 各患者多天血糖记录：(天数偏移, 测量类型, 血糖值)
GLUCOSE_SERIES = [
    ("patient_001", [
        (0, "空腹", 6.5), (0, "餐后", 8.2), (0, "睡前", 7.1),
        (1, "空腹", 7.8), (1, "餐后", 12.4), (1, "睡前", 9.5),
        (2, "空腹", 5.9), (2, "餐后", 9.0), (2, "睡前", 6.8),
        (3, "空腹", 3.4), (3, "餐后", 7.5), (3, "睡前", 6.2),
    ]),
    ("patient_002", [
        (0, "空腹", 5.2), (0, "餐后", 7.8), (1, "空腹", 5.6), (1, "餐后", 8.9),
    ]),
    ("patient_003", [
        (0, "睡前", 14.2), (1, "睡前", 11.8), (2, "睡前", 7.4),
    ]),
]

# 演示会话历史（问答 Agent）
CHAT_HISTORY = [
    ("patient_001", "多吃水果会不会升血糖？",
     "水果含果糖，同样会升血糖。建议在两餐之间少量食用低升糖指数水果（如苹果、柚子），"
     "并相应减少主食量，具体请结合您的血糖监测结果调整。"),
    ("patient_001", "我今天空腹血糖 7.8，是不是偏高了？",
     "您的空腹血糖 7.8 mmol/L 偏高（一般目标 4.4~7.0）。建议回顾近期饮食与用药是否规律，"
     "连续监测并复诊，必要时医生会调整用药方案。"),
]


async def _upsert_patient(session, data: dict) -> None:
    """按 patient_id 幂等地写入/更新患者档案。"""
    from sqlalchemy import select
    existing = await session.execute(
        select(PatientProfile).where(PatientProfile.patient_id == data["patient_id"])
    )
    row = existing.scalar_one_or_none()
    if row:
        row.name = data["name"]
        row.role = data["role"]
        row.discharge_summary = data["discharge_summary"]
        row.password_hash = hash_password(data["password"])
    else:
        session.add(PatientProfile(
            patient_id=data["patient_id"],
            password_hash=hash_password(data["password"]),
            role=data["role"],
            name=data["name"],
            discharge_summary=data["discharge_summary"],
        ))


async def seed_glucose(session) -> None:
    """写入多天血糖记录（含正常/偏高/偏低）。"""
    base = datetime.utcnow().replace(hour=8, minute=0, second=0, microsecond=0)
    for patient_id, records in GLUCOSE_SERIES:
        for day_offset, mtype, value in records:
            eag = mmol_to_mgdl(value)
            session.add(GlucoseRecord(
                patient_id=patient_id,
                value_mmol=value,
                measurement_type=mtype,
                eag_mgdl=round(eag, 1),
                hba1c_est=round(hba1c_from_eag(eag), 2),
                alert_level=classify_alert(value, mtype),
                recorded_at=base - timedelta(days=day_offset),
            ))


async def seed_chat_history(session) -> None:
    """写入多轮会话历史。"""
    for patient_id, q, a in CHAT_HISTORY:
        session.add(ChatHistory(
            patient_id=patient_id,
            session_id="demo_session",
            mode="qa",
            question=q,
            answer=a,
        ))


async def main():
    configure_logging("INFO")
    await init_db()

    async with get_session_factory()() as session:
        async with session.begin():
            for data in PATIENTS:
                await _upsert_patient(session, data)
            await _upsert_patient(session, DOCTOR)
            await seed_glucose(session)
            await seed_chat_history(session)

    logger.info("seed.done patients=%d", len(PATIENTS) + 1)
    print("演示数据初始化完成：3 名患者 + 1 名医生 + 血糖记录 + 会话历史。")


if __name__ == "__main__":
    asyncio.run(main())