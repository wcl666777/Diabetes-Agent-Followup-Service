"""意图识别 Agent 的状态定义（LangGraph StateGraph 的 StateSchema）。"""
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IntentState(TypedDict, total=False):
    # ① 消息历史（add_messages reducer：自动追加，不覆盖）
    messages: Annotated[list[BaseMessage], add_messages]

    # ② 请求上下文（编排器注入）
    patient_id: str
    session_id: str
    question: str

    # ③ 识别结果（classify 节点填充）
    question_type: str      # 饮食 / 运动 / 用药 / 血糖监测 / 并发症 / 生活方式 / 就医指导 / 综合咨询
    confidence: float       # 置信度（规则兜底时为 0）
    source: str             # 结果来源：llm（大模型）/ rule（规则兜底）
    structured_output: dict # 结构化输出（供编排器传递给下游 Agent）
