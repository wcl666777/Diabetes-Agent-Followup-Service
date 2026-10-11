"""三层兜底机制。

层次：自动重试 → Agent 级降级 → 系统级兜底。
即使底层（LLM / RAG / MySQL）全挂，也会返回一条友好提示，绝不把异常抛给用户。

用法：
    @with_retry(agent_type="qa")
    async def _invoke():
        return await graph.ainvoke(state, config=config)
"""
import asyncio
from functools import wraps
from typing import Any, Callable, Optional

from app.core.exceptions import (
    AuthenticationError,
    InvalidInputError,
    LLMAPIError,
    RAGConnectionError,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── 异常分类 ──
# 可重试：多半是短暂故障（网络抖动、超时），重试一下可能就好
RETRYABLE_ERRORS = (LLMAPIError, RAGConnectionError, TimeoutError, ConnectionError)
# 不可重试：重试也没用（输入非法、认证失败），应立即抛出
NON_RETRYABLE_ERRORS = (InvalidInputError, AuthenticationError)

MAX_RETRIES = 2                  # 最多重试 2 次（含首次共 3 次尝试）
RETRY_DELAYS = [1.0, 3.0]        # 第 1 次重试前等 1s，第 2 次前等 3s
TIMEOUT_PER_ATTEMPT = 120.0      # 单次调用超时（秒）


def with_retry(agent_type: str = ""):
    """三层兜底装饰器工厂。"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # ── 第一层：自动重试 ──
            last_error: Optional[Exception] = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), timeout=TIMEOUT_PER_ATTEMPT
                    )
                    if attempt > 0:
                        logger.info("retry.succeeded | agent=%s | attempt=%s", agent_type, attempt + 1)
                    return result
                except NON_RETRYABLE_ERRORS as e:
                    logger.warning("retry.non_retryable | agent=%s | error=%s", agent_type, e)
                    raise
                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = RETRY_DELAYS[attempt]
                        logger.warning(
                            "retry.attempt_failed | agent=%s | attempt=%s | delay=%s | error=%s",
                            agent_type, attempt + 1, delay, e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error("retry.all_failed | agent=%s | error=%s", agent_type, e)

            # ── 第二层：Agent 级降级 ──
            try:
                fallback = await AgentFallbackHandler.handle(agent_type, last_error)
                logger.info("retry.fallback_ok | agent=%s", agent_type)
                return fallback
            except Exception as fe:
                logger.error("retry.fallback_failed | agent=%s | error=%s", agent_type, fe)

            # ── 第三层：系统级兜底 ──
            logger.error("retry.system_fallback | agent=%s", agent_type)
            return _system_fallback_response(agent_type)

        return wrapper

    return decorator


class AgentFallbackHandler:
    """第二层降级：各 Agent 的专项降级策略（尽量保留核心功能）。"""

    @classmethod
    async def handle(cls, agent_type: str, original_error: Exception) -> Any:
        fallback_map = {
            "qa": cls._qa_fallback,
            "glucose": cls._glucose_fallback,
        }
        handler = fallback_map.get(agent_type)
        if handler:
            return await handler()
        raise original_error

    @classmethod
    async def _qa_fallback(cls) -> dict:
        """问答降级：知识库 / 模型不可用，提示稍后重试或联系医生。"""
        logger.info("fallback.qa_unavailable")
        return {
            "fallback_used": True,
            "content": "⚠️ 知识库检索暂时不可用，请稍后重试，或直接联系您的主管医生。",
            "structured_output": None,
        }

    @classmethod
    async def _glucose_fallback(cls) -> dict:
        """血糖管理降级：确定性计算服务不可用。"""
        logger.info("fallback.glucose_unavailable")
        return {
            "fallback_used": True,
            "content": "⚠️ 血糖记录服务暂时不可用，请稍后重新提交，您的数据未受影响。",
            "structured_output": None,
        }


def _system_fallback_response(agent_type: str) -> dict:
    """第三层：系统级兜底，永不失败。"""
    messages = {
        "qa": "非常抱歉，智能问答服务暂时不可用，请稍后再试，或直接联系您的主管医生。",
        "glucose": "非常抱歉，血糖管理服务暂时不可用，请稍后再试。",
    }
    content = messages.get(agent_type, "服务暂时不可用，请稍后再试。")
    return {
        "messages": [],
        "content": content,
        "fallback_used": True,
        "system_fallback": True,
        "structured_output": None,
    }
