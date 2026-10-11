"""统一异常体系。

所有自定义异常继承同一个基类，便于：
1. 「一次捕获全部」——上层用基类兜底；
2. 按「可重试 / 不可重试」分类处理（见 core/retry.py）。

在基类基础上，每个异常额外携带 agent_type（哪个 Agent 出错）与 details（细节字典）。
"""


class FollowUpBaseError(Exception):
    """所有随访助手自定义异常的基类。"""

    def __init__(self, message: str, agent_type: str = "", details: dict = None):
        super().__init__(message)
        self.agent_type = agent_type
        self.details = details or {}


class LLMAPIError(FollowUpBaseError):
    """大模型 API 调用失败（超时 / 限流 / 网络错误）。属于【可重试】异常。"""
    pass


class AgentExecutionError(FollowUpBaseError):
    """Agent 业务逻辑执行失败。"""
    pass


class PipelineError(FollowUpBaseError):
    """多 Agent Pipeline（流水线编排）失败。"""
    pass


class IntentRouteError(FollowUpBaseError):
    """意图识别路由失败（没判断出该交给哪个 Agent）。"""
    pass


class RAGConnectionError(FollowUpBaseError):
    """RAG / Milvus 检索服务连接失败。属于【可重试】异常。"""
    pass


class FileParseError(FollowUpBaseError):
    """文件解析失败（PDF / Word）。"""
    pass


class InvalidInputError(FollowUpBaseError):
    """用户输入不合法。属于【不可重试】异常（重试也不会变合法）。"""
    pass


class AuthenticationError(FollowUpBaseError):
    """认证失败。属于【不可重试】异常。"""
    pass
