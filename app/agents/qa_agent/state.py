"""问答 Agent 的状态定义（LangGraph StateGraph 的 StateSchema）。"""
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class QAState(TypedDict):
    # ① 消息历史（add_messages reducer：自动追加，不覆盖）
    messages: Annotated[list[BaseMessage], add_messages]

    # ② 请求上下文（API 层注入）
    patient_id: str
    session_id: str
    question: str
    force_human: bool                  # 患者主动要求人工：跳过检索直接转人工
    retrieval_precomputed: bool        # 上游接口已完成检索并注入结果：节点不再重复检索

    # ③ 检索结果（retrieve 节点填充）
    confidence: float                  # 精排置信度
    personal_context: str              # 患者个人资料（病例 / 出院小结）文本
    is_high_risk: bool                 # 是否高危问题
    query_category: str                # 意图识别：通用知识 / 专业咨询
    strategy: str                      # 检索策略：直接 / 假设 / 子查询 / 回溯
    context: str                       # RAG 侧融合好的上下文（公共指南 + 个人资料）
    source: str                        # 答案来源：faq（快车道）/ rag / none
    faq_answer: str                    # 命中快车道时的标准答案

    # ④ 人工介入（human_review 节点填充）
    need_human: bool                   # 是否需要人工介入
    doctor_note: str                   # 医生补充意见（resume 时带入）

    # ⑤ 生成结果
    answer: str
    sources: list[str]
    answer_mode: str                   # "rag"（AI 生成）/ "human"（医生直接回答）
    question_type: str                 # 细粒度问题类型：用药 / 饮食 / 血糖监测 / 并发症 / 生活方式 / 就医指导 / 综合咨询