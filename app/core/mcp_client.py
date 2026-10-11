"""MCP 客户端：通过 Streamable HTTP + JSON-RPC 调用 RAG 服务的 MCP Tool。

RAG 服务对外暴露 FastMCP Server（stateless，每次调用自包含、无需握手），
本模块直接 POST "tools/call" 到 MCP 端点。
"""
import json
from typing import Any

import httpx
import tenacity

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _endpoint() -> str:
    """MCP 端点：rag_mcp_server_url 为 base（如 http://localhost:8001/rag），Streamable HTTP 端点在 /mcp。"""
    return f"{get_settings().rag_mcp_server_url.rstrip('/')}/mcp"


@tenacity.retry(
    wait=tenacity.wait_fixed(0.5),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
    retry=tenacity.retry_if_exception_type((httpx.HTTPError, ValueError)),
)
async def call_mcp_tool(tool_name: str, arguments: dict[str, Any], timeout: float = 30.0) -> Any:
    """调用 RAG MCP Server 的单个工具，返回工具结果（JSON 反序列化后）。"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",  # json_response=True 的 Server 要求此头
    }

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(_endpoint(), json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    if "error" in data:
        raise ValueError(f"MCP tool '{tool_name}' error: {data['error'].get('message')}")

    content = data.get("result", {}).get("content", [])
    if not content:
        return []

    items = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = text
        if isinstance(parsed, list):
            return parsed  # 整个列表在一个 TextContent 里（FastMCP 常见行为）
        items.append(parsed)

    return items if items else content


async def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    """对 RAG 知识库做混合检索（透过 MCP），返回精排文档列表。"""
    logger.info("mcp.search_knowledge_base | query=%s", query[:30])
    return await call_mcp_tool("search_knowledge_base", {"query": query, "top_k": top_k})


async def rag_query(query: str, personal_context: str = "") -> dict:
    """走 RAG 主流程备料（缓存 → MySQL 快车道 → 意图识别 → 检索策略 → 融合个人资料）。

    Returns:
        {source, answer, context, confidence, query_category, strategy, sources}
        - source="faq"：answer 为标准答案，Agent 直接输出；
        - source="rag"/"none"：context 为融合好的上下文，Agent 用它流式生成。
    """
    logger.info("mcp.rag_query | query=%s", query[:30])
    result = await call_mcp_tool(
        "rag_query",
        {"query": query, "personal_context": personal_context},
        timeout=90.0,  # 策略检索可能触发多次 LLM + 检索，放宽超时
    )
    if isinstance(result, list) and result:
        return result[0] if isinstance(result[0], dict) else {}
    return result if isinstance(result, dict) else {}