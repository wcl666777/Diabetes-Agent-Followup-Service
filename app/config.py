"""Agent 服务配置中心。

所有配置从 .env 读取（pydantic-settings 自动完成类型转换与校验）。
用法：from app.config import get_settings; settings = get_settings()
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── 服务监听 ──
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    log_dir: str = "./logs"             # 日志落盘目录（相对服务启动目录）

    # ── 大模型（DeepSeek，OpenAI 兼容）──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # ── MySQL（患者档案 / 血糖记录 / 会话历史 / 人工介入）──
    db_host: str = "localhost"
    db_port: int = 3307
    db_name: str = "diabetes"
    db_user: str = "diabetes"
    db_password: str = "diabetes123"

    # ── JWT 认证 ──
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION_use_a_long_random_string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60      # 访问令牌有效期（分钟）
    refresh_token_expire_days: int = 7         # 刷新令牌有效期（天）

    # ── RAG 服务的 MCP Server 地址（问答 Agent 通过 MCP 调用知识检索）──
    rag_mcp_server_url: str = "http://localhost:8001/rag"

    # ── 问答 Agent 转人工阈值 ──
    qa_confidence_threshold: float = 0.5   # 精排置信度低于此值 → 转人工

    @property
    def database_url(self) -> str:
        # aiomysql 为 SQLAlchemy 的 MySQL 异步驱动，适配 FastAPI 异步生态
        return (
            f"mysql+aiomysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()