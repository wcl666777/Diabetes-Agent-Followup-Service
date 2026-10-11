"""血糖管理 Agent 的节点实现：记录 → 解读（确定性计算 + 告警）。

所有医学指标均由 calculator 计算，LLM 不参与数值计算。
"""
from langchain_core.messages import AIMessage

from app.agents.glucose_agent import calculator
from app.agents.glucose_agent.state import GlucoseState
from app.core.logger import get_logger

logger = get_logger(__name__)


async def _fetch_recent_values(patient_id: str, limit: int = 13) -> list[float]:
    """取患者最近 limit 条血糖值（不含本次），DB 不可用则返回空。"""
    try:
        from sqlalchemy import select
        from app.core.db import get_session_factory
        from app.core.models import GlucoseRecord

        async with get_session_factory()() as session:
            result = await session.execute(
                select(GlucoseRecord.value_mmol)
                .where(GlucoseRecord.patient_id == patient_id)
                .order_by(GlucoseRecord.recorded_at.desc())
                .limit(limit)
            )
            return [round(row[0], 2) for row in result.all()]
    except Exception as e:
        logger.warning("glucose.fetch_history_failed | error=%s", e)
        return []


async def _persist(state: GlucoseState, metrics: dict, alert_level: str) -> bool:
    """落库本次血糖记录。失败仅告警，不阻断解读。"""
    try:
        from app.core.db import get_session_factory
        from app.core.models import GlucoseRecord

        async with get_session_factory()() as session:
            session.add(
                GlucoseRecord(
                    patient_id=state["patient_id"],
                    value_mmol=state["value_mmol"],
                    measurement_type=state["measurement_type"],
                    eag_mgdl=metrics["eag_mgdl"],
                    hba1c_est=metrics["hba1c_est"],
                    alert_level=alert_level,
                )
            )
            await session.commit()
        return True
    except Exception as e:
        logger.warning("glucose.persist_failed | error=%s", e)
        return False


async def record_node(state: GlucoseState) -> dict:
    """记录节点：分类告警 → 计算指标（含历史序列 TIR/CV 等）→ 落库。"""
    value = state["value_mmol"]
    mtype = state["measurement_type"]

    alert_level = calculator.classify_alert(value, mtype)

    # 单次值 → eAG / 反推 HbA1c
    eag = calculator.mmol_to_mgdl(value)
    hba1c_est = calculator.hba1c_from_eag(eag)

    # 历史序列（含本次）→ TIR / TAR / TBR / CV
    history = await _fetch_recent_values(state["patient_id"])
    series = history + [value]
    metrics = calculator.compute_series_metrics(series)

    metrics["eag_current_mgdl"] = round(eag, 1)          # 本次单次的 eAG
    metrics["hba1c_est_current"] = round(hba1c_est, 2)   # 本次单次反推 HbA1c

    saved = await _persist(state, metrics, alert_level)

    logger.info(
        "glucose.record_done | value=%.1f | type=%s | alert=%s",
        value, mtype, alert_level,
    )
    return {"alert_level": alert_level, "metrics": metrics, "saved": saved}


def _interpret_advice(state: GlucoseState) -> dict:
    """确定性解读 + 建议（依据 alert_level 与指标生成模板文案）。"""
    value = state["value_mmol"]
    mtype = state["measurement_type"]
    alert = state["alert_level"]
    m = state["metrics"]

    if alert == "low":
        interpretation = (
            f"本次血糖 {value:.1f} mmol/L，低于 3.9 mmol/L，属于低血糖，风险较高。"
            "若意识清醒，请立即按 15-15 原则补充快速碳水（如半杯果汁、3~4 块方糖），"
            "15 分钟后复测；若出现意识不清，请立即拨打急救电话。"
        )
        advice = "低血糖缓解后 1 小时内避免剧烈运动；建议排查是否用药过量或进食不足。"
    elif alert == "high":
        interpretation = (
            f"本次血糖 {value:.1f} mmol/L（{mtype}），明显偏高。"
            "若同时伴恶心、呕吐、呼吸深快或口渴多尿加重，请警惕糖尿病酮症酸中毒并尽快就医。"
        )
        advice = "多饮水、避免高糖饮食；若持续偏高，请及时复诊调整方案。"
    else:
        interpretation = (
            f"本次血糖 {value:.1f} mmol/L（{mtype}），在目标范围内，控制良好。"
        )
        advice = "保持规律作息与定时定量进餐，继续按时监测血糖。"

    # 附上序列指标（供患者了解整体控制情况）
    series_note = (
        f"近 {m['count']} 次血糖指标：均值 {m['mean_mmol']} mmol/L，"
        f"反推 HbA1c 约 {m['hba1c_est']}%，TIR {m['tir']}%，TAR {m['tar']}%，"
        f"TBR {m['tbr']}%，变异系数 CV {m['cv']}%。"
    )
    return {"interpretation": interpretation + " " + series_note, "advice": advice}


async def interpret_node(state: GlucoseState) -> dict:
    """解读节点：生成解读与建议，并输出最终回答消息。"""
    result = _interpret_advice(state)
    answer = result["interpretation"] + "\n建议：" + result["advice"]

    if state["alert_level"] in ("high", "low"):
        answer = "⚠️ " + answer

    return {**result, "messages": [AIMessage(content=answer)]}