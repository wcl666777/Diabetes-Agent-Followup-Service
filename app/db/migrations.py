"""数据库幂等迁移。

定位：init_db() 用 create_all 建「缺失的表」；本模块负责给「已存在的表」补
新列 / 新索引。两者都幂等，可反复执行。

注意：MySQL 不支持 `ADD COLUMN IF NOT EXISTS`（那是 MariaDB/PG 语法），
因此这里先查 information_schema 判断是否已存在，再决定是否执行 DDL。
"""
from sqlalchemy import text

from app.core.db import get_engine
from app.core.logger import get_logger

logger = get_logger(__name__)

# 待补的索引：(表名, 索引名, 列定义)
_MIGRATIONS_INDEX = [
    ("chat_history", "idx_chat_patient_created", "patient_id, created_at"),
    ("glucose_records", "idx_glucose_patient_recorded", "patient_id, recorded_at"),
    ("patient_documents", "idx_doc_patient_type", "patient_id, doc_type"),
    ("patient_doc_chunks", "idx_chunk_patient_level", "patient_id, level"),
    ("human_sessions", "idx_human_session_patient_status", "patient_id, status"),
    ("human_messages", "idx_human_message_patient", "patient_id, id"),
]

# 待补的列：(表名, 列名, 列定义) —— 为已存在的表补新字段
_MIGRATIONS_COLUMN = [
    ("hitl_pending", "status", "VARCHAR(16) NOT NULL DEFAULT 'pending'"),
    ("hitl_pending", "answer", "TEXT"),
]


async def _index_exists(conn, table: str, index: str) -> bool:
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = :t AND index_name = :i"
        ),
        {"t": table, "i": index},
    )
    return (result.scalar() or 0) > 0


async def _table_exists(conn, table: str) -> bool:
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = :t"
        ),
        {"t": table},
    )
    return (result.scalar() or 0) > 0


async def _column_exists(conn, table: str, column: str) -> bool:
    result = await conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    )
    return (result.scalar() or 0) > 0


async def run_migrations() -> None:
    """执行幂等迁移：为已存在的表补索引 / 补列。失败仅告警，不阻断启动。"""
    engine = get_engine()
    applied = 0
    try:
        async with engine.begin() as conn:
            for table, index, columns in _MIGRATIONS_INDEX:
                if not await _table_exists(conn, table):
                    continue
                if await _index_exists(conn, table, index):
                    continue
                await conn.execute(text(f"CREATE INDEX {index} ON {table} ({columns})"))
                applied += 1
                logger.info("migrations.index_created | %s.%s", table, index)

            for table, column, definition in _MIGRATIONS_COLUMN:
                if not await _table_exists(conn, table):
                    continue
                if await _column_exists(conn, table, column):
                    continue
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
                applied += 1
                logger.info("migrations.column_added | %s.%s", table, column)
    except Exception as e:
        logger.warning("migrations.failed | error=%s", e)
        return
    logger.info("migrations.done | applied=%d", applied)
