"""Agent 服务入口：FastAPI + 双 Agent 路由 + 认证 + 静态前端。

启动：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
前端：http://localhost:8000/
API 文档：http://localhost:8000/docs
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, chat, glucose, hitl, patient_docs, unified_chat
from app.config import get_settings
from app.core.logger import configure_logging, get_logger, set_trace_id

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level, settings.log_dir)
    logger = get_logger(__name__)
    logger.info("agent_service.starting | port=%s", settings.app_port)

    # 建表（幂等）；MySQL 未就绪时仅告警，不拦启动
    try:
        from app.core.db import init_db
        await init_db()
        # 建表后再跑幂等迁移（给已有表补索引）
        from app.db.migrations import run_migrations
        await run_migrations()
    except Exception as e:
        logger.warning("agent_service.db_init_failed | error=%s", e)

    logger.info("agent_service.started")
    yield

    from app.core.db import dispose_db
    await dispose_db()
    logger.info("agent_service.shutting_down")


app = FastAPI(
    title="糖尿病出院患者智能随访助手（Agent）",
    description="基于 LangGraph 的双 Agent：问答 Agent（RAG + 人工介入）+ 血糖管理 Agent（确定性计算 + 告警），"
                "内置 JWT 认证与 MySQL 持久化。",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """为每个请求生成/透传 trace_id，贯穿结构化日志。"""
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    set_trace_id(trace_id)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(glucose.router)
app.include_router(hitl.router)
app.include_router(patient_docs.router)
app.include_router(unified_chat.router)

# 静态前端
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/health", tags=["系统"])
async def health_check():
    return {"status": "ok", "service": "agent_service"}