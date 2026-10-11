"""LangGraph 记忆管理：每个 Agent 独立的 MemorySaver + thread_id 构造。

MemorySaver 是 LangGraph 的内存 checkpointer：
  - 多轮对话记忆：同一 thread_id 复用历史
  - Human-in-the-Loop：interrupt() 暂停时状态被持久化，resume 用同一 thread_id 恢复

生产可替换为 AsyncPostgresSaver 持久化，业务代码无需改动。
"""
from langgraph.checkpoint.memory import MemorySaver

from app.core.logger import get_logger

logger = get_logger(__name__)

_memory_savers: dict[str, MemorySaver] = {}


def get_memory_saver(agent_type: str = "default") -> MemorySaver:
    """获取指定 Agent 的 MemorySaver 单例（不同 Agent schema 不同，必须隔离）。"""
    if agent_type not in _memory_savers:
        _memory_savers[agent_type] = MemorySaver()
        logger.info("memory.saver_initialized | agent=%s", agent_type)
    return _memory_savers[agent_type]


def build_thread_id(patient_id: str, session_id: str) -> str:
    """构造 checkpointer 使用的 thread_id（患者 + 会话隔离）。"""
    return f"patient_{patient_id}_session_{session_id}"


def build_config(patient_id: str, session_id: str) -> dict:
    """构造 LangGraph 调用 config。"""
    return {"configurable": {"thread_id": build_thread_id(patient_id, session_id)}}