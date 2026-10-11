"""血糖计算模块（calculator.py）单元测试。

覆盖：单位换算、糖化/平均血糖互推、告警分级边界值、时间序列指标。
纯函数测试，无任何外部依赖，可直接运行：pytest tests/test_calculator.py
"""
import math

import pytest

from app.agents.glucose_agent.calculator import (
    MMOL_TO_MGDL,
    classify_alert,
    compute_series_metrics,
    eag_from_hba1c,
    hba1c_from_eag,
    mgdl_to_mmol,
    mmol_to_mgdl,
)


class TestUnitConversion:
    def test_mmol_to_mgdl(self):
        assert math.isclose(mmol_to_mgdl(5.0), 5.0 * MMOL_TO_MGDL, rel_tol=1e-9)

    def test_mgdl_to_mmol_roundtrip(self):
        mgdl = mmol_to_mgdl(6.5)
        assert math.isclose(mgdl_to_mmol(mgdl), 6.5, rel_tol=1e-9)

    def test_zero(self):
        assert mmol_to_mgdl(0.0) == 0.0


class TestHbA1cConversion:
    def test_eag_from_hba1c(self):
        # ADA 公式：eAG = 28.7 * HbA1c - 46.7
        assert math.isclose(eag_from_hba1c(7.0), 28.7 * 7.0 - 46.7, rel_tol=1e-9)

    def test_hba1c_from_eag(self):
        assert math.isclose(hba1c_from_eag(154.2), 7.0, rel_tol=1e-6)

    def test_roundtrip(self):
        hba1c = 6.5
        assert math.isclose(hba1c_from_eag(eag_from_hba1c(hba1c)), hba1c, rel_tol=1e-9)


class TestClassifyAlert:
    def test_low_below_39(self):
        assert classify_alert(3.8, "空腹") == "low"
        assert classify_alert(3.0, "餐后") == "low"

    def test_fasting_high(self):
        assert classify_alert(7.0, "空腹") == "high"   # 边界：≥7.0
        assert classify_alert(7.5, "空腹") == "high"

    def test_fasting_normal_boundary(self):
        assert classify_alert(6.99, "空腹") == "normal"

    def test_postprandial_high(self):
        assert classify_alert(11.1, "餐后") == "high"  # 边界：≥11.1
        assert classify_alert(12.0, "睡前") == "high"

    def test_postprandial_normal(self):
        assert classify_alert(10.9, "餐后") == "normal"

    def test_normal(self):
        assert classify_alert(5.5, "空腹") == "normal"
        assert classify_alert(8.0, "餐后") == "normal"


class TestComputeSeriesMetrics:
    def test_empty_series(self):
        m = compute_series_metrics([])
        assert m["count"] == 0
        assert m["tir"] == 0.0 and m["tar"] == 0.0 and m["tbr"] == 0.0
        assert m["mean_mmol"] == 0.0

    def test_all_in_range(self):
        series = [5.0, 6.0, 7.0, 8.0]  # 全在 3.9~10.0
        m = compute_series_metrics(series)
        assert m["tir"] == 100.0
        assert m["tar"] == 0.0
        assert m["tbr"] == 0.0
        assert m["count"] == 4

    def test_mixed_series(self):
        series = [12.0, 5.0, 3.0, 6.0]  # 1 高、1 低、2 正常
        m = compute_series_metrics(series)
        assert m["tar"] == 25.0   # 1/4
        assert m["tbr"] == 25.0   # 1/4
        assert m["tir"] == 50.0   # 2/4

    def test_cv_zero_when_constant(self):
        m = compute_series_metrics([5.0, 5.0, 5.0])
        assert m["cv"] == 0.0

    def test_mean_and_eag(self):
        series = [5.0, 6.0]
        m = compute_series_metrics(series)
        assert math.isclose(m["mean_mmol"], 5.5, rel_tol=1e-9)
        # eag_mgdl 在函数内做了 round(eag, 1)；hba1c_est 用未取整的 eag 计算后再 round 2 位
        eag_unrounded = mmol_to_mgdl(5.5)
        assert math.isclose(m["eag_mgdl"], round(eag_unrounded, 1), rel_tol=1e-9)
        assert math.isclose(m["hba1c_est"], round(hba1c_from_eag(eag_unrounded), 2), rel_tol=1e-9)