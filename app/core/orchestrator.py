"""多 Agent 编排器。

职责：
1. 按 AgentType 懒加载并缓存对应的 LangGraph 图（首次调用才 import + build）；
2. 统一入口 handle()：把 AgentRequest 转为图输入、用三层兜底执行、归一化输出；
3. 任何异常都被兜底成友好响应，绝不向上抛。

统一输出约定：从结果 state 的最后一条消息取正文，兼容 qa / glucose。
"""
from enum import Enum
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.core.memory import build_thread_id
from app.core.question_types import classify_question_type
from app.core.retry import with_retry

logger = get_logger(__name__)


class AgentType(str, Enum):
    """可编排的 Agent 类型。"""

    INTENT = "intent"    # 意图识别 Agent（问答前置：识别饮食 / 运动 / 用药…）
    QA = "qa"            # 问答 Agent（RAG + 人工介入）
    GLUCOSE = "glucose"  # 血糖管理 Agent（确定性计算 + 告警）


class AgentRequest(BaseModel):
    """统一的 Agent 调用请求。"""

    patient_id: str
    session_id: str
    agent_type: str = AgentType.QA
    user_message: str = ""
    context: dict = Field(default_factory=dict, description="额外初始状态（如血糖值 / 类型）")

    @property
    def thread_id(self) -> str:
        return build_thread_id(self.patient_id, self.session_id)


class AgentResponse(BaseModel):
    """统一的 Agent 调用响应。"""

    success: bool
    agent_type: str
    content: str = ""
    structured: Optional[dict] = None
    fallback_used: bool = False
    error_msg: str = ""
    metadata: dict = Field(default_factory=dict)


def _last_message_text(state: dict) -> str:
    """从结果 state 的最后一条消息中提取文本（兼容 str / list 内容）。"""
    messages = state.get("messages") or []
    if not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


class Orchestrator:
    """Agent 编排器（单例使用）。"""

    def __init__(self) -> None:
        self._agent_graphs: Dict[str, Any] = {}

    def _get_agent_graph(self, agent_type: str):
        """按类型懒加载并缓存 Agent 图。"""
        if agent_type not in self._agent_graphs:
            if agent_type == AgentType.INTENT:
                from app.agents.intent_agent.graph import build_intent_graph
                graph = build_intent_graph()
            elif agent_type == AgentType.QA:
                from app.agents.qa_agent.graph import build_qa_graph
                graph = build_qa_graph()
            elif agent_type == AgentType.GLUCOSE:
                from app.agents.glucose_agent.graph import build_glucose_graph
                graph = build_glucose_graph()
            else:
                raise ValueError(f"未知 agent_type: {agent_type}")
            self._agent_graphs[agent_type] = graph
            logger.info("orchestrator.graph_loaded | agent=%s", agent_type)
        return self._agent_graphs[agent_type]

    async def handle(self, request: AgentRequest) -> AgentResponse:
        """统一入口：异常兜底，永远返回 AgentResponse。"""
        try:
            return await self._run_single_agent(request)
        except Exception as e:
            logger.error("orchestrator.handle_failed | agent=%s | error=%s", request.agent_type, e)
            return AgentResponse(
                success=False,
                agent_type=request.agent_type,
                content="系统处理请求时遇到问题，请稍后重试。",
                error_msg=str(e),
            )

    async def run_intent(self, patient_id: str, session_id: str, question: str) -> dict:
        """编排第一步：跑「意图识别 Agent」，返回识别结果供「问答 Agent」使用。

        这是「意图识别 Agent → 问答 Agent」两步串联的第一环：
        question_type（饮食 / 运动 / 用药…）会随请求传入问答 Agent，使其生成更聚焦。

        返回：{"question_type": str, "confidence": float, "source": "llm"/"rule"}
        """
        request = AgentRequest(
            patient_id=patient_id,
            session_id=session_id,
            agent_type=AgentType.INTENT,
            user_message=question,
            context={"question": question},  # 意图图读取 state["question"]
        )
        response = await self.handle(request)
        if response.structured:
            return response.structured
        # 兜底：意图 Agent 整体降级时，退回规则分类，保证调用方总能拿到类型
        return {
            "question_type": classify_question_type(question),
            "confidence": 0.0,
            "source": "rule",
        }

    async def _run_single_agent(self, request: AgentRequest) -> AgentResponse:
        """执行单个 Agent（含三层兜底）。"""
        graph = self._get_agent_graph(request.agent_type)

        initial_state = {
            "messages": [HumanMessage(content=request.user_message)],
            "patient_id": request.patient_id,
            "session_id": request.session_id,
            **request.context,
        }
        config = {"configurable": {"thread_id": request.thread_id}}

        async def _invoke():
            return await graph.ainvoke(initial_state, config=config)

        result = await with_retry(agent_type=request.agent_type)(_invoke)()

        # 降级 / 系统兜底结果（非 state，而是 fallback dict）
        if isinstance(result, dict) and result.get("fallback_used"):
            return AgentResponse(
                success=True,
                agent_type=request.agent_type,
                content=result.get("content", ""),
                fallback_used=True,
                metadata=result,
            )

        content = _last_message_text(result) or result.get("answer", "")
        return AgentResponse(
            success=True,
            agent_type=request.agent_type,
            content=content,
            structured=result.get("structured_output"),
            metadata={
                "answer_mode": result.get("answer_mode", ""),
                "confidence": result.get("confidence"),
                "sources": result.get("sources", []),
                "alert_level": result.get("alert_level", ""),
            },
        )


_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """编排器单例。"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
