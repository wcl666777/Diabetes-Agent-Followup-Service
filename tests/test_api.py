"""认证接口集成测试（注册 / 登录 / 刷新令牌）。

依赖：MySQL 已启动且已建表（docker compose up -d mysql && python scripts/seed_patient.py）。
未就绪时自动跳过，不影响单元测试。

运行：pytest tests/test_api.py
"""
import time

import httpx
import pytest

from app.config import get_settings
from app.core.db import dispose_db, init_db
from app.main import app


def _db_ready() -> bool:
    """用同步驱动 pymysql 探测 MySQL 是否可用（避免引入额外事件循环）。"""
    try:
        import pymysql

        s = get_settings()
        conn = pymysql.connect(
            host=s.db_host,
            port=s.db_port,
            user=s.db_user,
            password=s.db_password,
            database=s.db_name,
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(not _db_ready(), reason="MySQL 未就绪，跳过集成测试")


@requires_db
@pytest.mark.asyncio
async def test_register_login_refresh_flow():
    """完整认证链路：注册 → 登录 → 刷新令牌（单测试内统一事件循环）。"""
    await init_db()

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1) 注册（使用临时账号，避免与已有数据冲突）
            patient_id = f"test_{int(time.time() * 1000)}"
            r = await client.post(
                "/api/v1/auth/register",
                json={"patient_id": patient_id, "password": "test123456", "name": "集成测试"},
            )
            assert r.status_code == 201, r.text
            data = r.json()
            assert data["patient_id"] == patient_id
            assert data["role"] == "patient"
            assert data["access_token"] and data["refresh_token"]

            # 2) 重复注册应返回 409
            r = await client.post(
                "/api/v1/auth/register",
                json={"patient_id": patient_id, "password": "test123456"},
            )
            assert r.status_code == 409

            # 3) 登录（正确密码）
            r = await client.post(
                "/api/v1/auth/login",
                json={"patient_id": patient_id, "password": "test123456"},
            )
            assert r.status_code == 200, r.text

            # 4) 登录（错误密码）应返回 401
            r = await client.post(
                "/api/v1/auth/login",
                json={"patient_id": patient_id, "password": "wrong-password"},
            )
            assert r.status_code == 401

            # 5) 刷新令牌（同一秒内 iat/exp 相同可能导致 token 字符串一致，属正常 JWT 行为）
            refresh = data["refresh_token"]
            r = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
            assert r.status_code == 200, r.text
            assert r.json()["access_token"]
            assert r.json()["patient_id"] == patient_id
    finally:
        await dispose_db()


@requires_db
@pytest.mark.asyncio
async def test_chat_requires_auth():
    """未携带令牌访问业务接口应返回 401（此路径不触发 DB 连接）。"""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/v1/chat", json={"question": "出院后能吃什么？"})
    assert r.status_code == 401