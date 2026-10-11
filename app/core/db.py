"""数据库连接（SQLAlchemy 异步 + aiomysql）。

提供异步 engine、session 工厂与 Base。启动时 init_db() 建表（幂等）。
生产环境建议用 Alembic 做迁移，本项目为控制复杂度采用 create_all 幂等建表。
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        # pool_recycle：在 MySQL wait_timeout 之前回收连接，避免拿到失效连接。
        # 注意：aiomysql 下 pool_pre_ping 存在 ping() 缺参兼容问题，故不用 pre_ping。
        _engine = create_async_engine(
            get_settings().database_url,
            echo=False,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def init_db() -> None:
    """建表（幂等）。首次启动或模型变更时调用。"""
    from app.core import models  # noqa: F401  确保模型已注册到 Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db.init_done")


async def dispose_db() -> None:
    """关闭引擎连接池（服务优雅退出时调用）。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None