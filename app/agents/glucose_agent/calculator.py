"""血糖管理 Agent 的确定性医学计算模块。

核心原则（项目技术亮点）：医学指标一律用代码计算，不交给 LLM，
避免大模型数值不可靠带来的医疗风险。
"""
from typing import List

# 单位换算常数
MMOL_TO_MGDL = 18.0182


def mmol_to_mgdl(value_mmol: float) -> float:
    """mmol/L → mg/dL。"""
    return value_mmol * MMOL_TO_MGDL


def mgdl_to_mmol(value_mgdl: float) -> float:
    """mg/dL → mmol/L。"""
    return value_mgdl / MMOL_TO_MGDL


def eag_from_hba1c(hba1c: float) -> float:
    """由糖化血红蛋白推平均血糖（美国 ADA 公式）：
    eAG(mg/dL) = 28.7 × HbA1c(%) − 46.7
    """
    return 28.7 * hba1c - 46.7


def hba1c_from_eag(eag_mgdl: float) -> float:
    """由平均血糖反推 HbA1c：
    HbA1c(%) = (eAG + 46.7) / 28.7
    """
    return (eag_mgdl + 46.7) / 28.7


def classify_alert(value_mmol: float, measurement_type: str = "空腹") -> str:
    """对单次血糖值给出告警级别：normal / high / low。

    参考阈值（一般成人）：
      - < 3.9            → 低血糖
      - 空腹 ≥ 7.0       → 偏高
      - 餐后/随机 ≥ 11.1 → 偏高
    """
    if value_mmol < 3.9:
        return "low"
    if measurement_type == "空腹":
        if value_mmol >= 7.0:
            return "high"
    else:
        if value_mmol >= 11.1:
            return "high"
    return "normal"


def compute_series_metrics(series: List[float]) -> dict:
    """由血糖时间序列计算 TIR / TAR / TBR / CV / 均值 / eAG / 反推 HbA1c。

    标准（CGM 目标）：
      - TIR：3.9~10.0 mmol/L 的时间占比，目标 > 70%
      - TAR：> 10.0 的时间占比
      - TBR：< 3.9 的时间占比
      - CV：变异系数 = 标准差 / 均值
    """
    if not series:
        return {"tir": 0.0, "tar": 0.0, "tbr": 0.0, "cv": 0.0,
                "mean_mmol": 0.0, "eag_mgdl": 0.0, "hba1c_est": 0.0, "count": 0}

    n = len(series)
    in_range = sum(1 for v in series if 3.9 <= v <= 10.0)
    above = sum(1 for v in series if v > 10.0)
    below = sum(1 for v in series if v < 3.9)

    mean = sum(series) / n
    variance = sum((v - mean) ** 2 for v in series) / n
    std = variance ** 0.5
    cv = (std / mean * 100) if mean else 0.0

    eag = mmol_to_mgdl(mean)
    return {
        "tir": round(in_range / n * 100, 1),
        "tar": round(above / n * 100, 1),
        "tbr": round(below / n * 100, 1),
        "cv": round(cv, 1),
        "mean_mmol": round(mean, 2),
        "eag_mgdl": round(eag, 1),
        "hba1c_est": round(hba1c_from_eag(eag), 2),
        "count": n,
    }