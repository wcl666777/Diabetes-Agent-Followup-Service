"""DeepSeek 大模型工厂（统一封装，供 LangGraph Agent 使用）。

规矩：所有 Agent 通过 get_llm() 获取模型，禁止直接 new 模型。
DeepSeek 支持 OpenAI 兼容接口，用 langchain 的 init_chat_model 加载。
"""
from typing import Any, Dict, Type

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# 绕过系统代理：DeepSeek 国内可直连，避免被代理劫持导致 TLS 失败
_HTTP_ASYNC_CLIENT = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(120.0, connect=15.0))
_HTTP_SYNC_CLIENT = httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0, connect=15.0))

# Agent 类型 → 模型标识符（想换模型只改这里）
_AGENT_MODEL_ROUTING: Dict[str, str] = {
    "qa": "deepseek-chat",       # 问答 Agent
    "glucose": "deepseek-chat",  # 血糖管理 Agent
    "intent": "deepseek-chat",   # 统一入口的意图路由
}

_instances: Dict[str, BaseChatModel] = {}


def _build_kwargs(model_name: str) -> Dict[str, Any]:
    s = get_settings()
    return {
        "model": s.llm_model,
        "model_provider": "openai",
        "temperature": 0,
        "api_key": s.deepseek_api_key,
        "base_url": s.deepseek_base_url,
        "max_retries": 2,  # LLM 调用自动重试（OpenAI SDK 内部基于 tenacity）
        "http_async_client": _HTTP_ASYNC_CLIENT,
        "http_client": _HTTP_SYNC_CLIENT,
    }


def get_llm(agent_type: str, temperature: float = 0, streaming: bool = False) -> BaseChatModel:
    """按 Agent 类型获取模型（带缓存）。"""
    if agent_type not in _AGENT_MODEL_ROUTING:
        raise ValueError(f"未知 agent_type: {agent_type}，可用：{list(_AGENT_MODEL_ROUTING)}")

    cache_key = f"{agent_type}_{temperature}_{streaming}"
    if cache_key not in _instances:
        kwargs = _build_kwargs(agent_type)
        kwargs["temperature"] = temperature
        kwargs["streaming"] = streaming
        _instances[cache_key] = init_chat_model(**kwargs)
        logger.info("llm_factory.initialized | agent=%s", agent_type)
    return _instances[cache_key]


def get_structured_llm(agent_type: str, output_schema: Type[BaseModel]) -> Runnable:
    """获取绑定结构化输出的模型。"""
    llm = get_llm(agent_type)
    return llm.with_structured_output(output_schema, method="function_calling")