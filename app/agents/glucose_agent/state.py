"""血糖管理 Agent 的状态定义。"""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GlucoseState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # 请求上下文
    patient_id: str
    session_id: str
    value_mmol: float                    # 本次上报血糖值
    measurement_type: str                # 空腹 / 餐后 / 睡前 / 随机

    # 计算与解读结果
    alert_level: str                     # normal / high / low
    metrics: dict                        # TIR/TAR/TBR/CV/eAG/HbA1c 等
    interpretation: str                  # 解读文本
    advice: str                          # 生活方式建议
    saved: bool                          # 是否成功落库