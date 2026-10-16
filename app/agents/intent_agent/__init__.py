"""意图识别 Agent：判断患者问题属于哪一类（饮食 / 运动 / 用药 ...）。

定位：编排在「问答 Agent 之前」，识别结果（question_type）作为问答 Agent 的输入，
让生成回答时更聚焦。LLM 主导 + 规则兜底，永不因模型异常而失败。
"""
from app.agents.intent_agent.graph import build_intent_graph
from app.agents.intent_agent.nodes import classify_intent

__all__ = ["build_intent_graph", "classify_intent"]
