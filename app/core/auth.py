"""JWT 认证与密码哈希。

PyJWT 签发/校验令牌，bcrypt 做密码哈希。
不引入 passlib：passlib 已停止维护且与 bcrypt>=4 存在兼容问题，直接用 bcrypt 更可控。
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select

from app.config import get_settings
from app.core.db import get_session_factory
from app.core.models import PatientProfile

# tokenUrl 指向登录接口的 OpenAPI 描述（Swagger 授权按钮用）
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── 密码哈希 ──
def hash_password(plain: str) -> str:
    """用 bcrypt 对明文密码做加盐哈希。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码与哈希是否匹配。"""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── 令牌签发 / 校验 ──
def create_access_token(subject: str, role: str) -> str:
    """签发访问令牌（短时效）。subject 取 patient_id。"""
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=s.access_token_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def create_refresh_token(subject: str, role: str) -> str:
    """签发刷新令牌（长时效，用来无感续期）。"""
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=s.refresh_token_expire_days),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str, expected_type: str = "access") -> dict:
    """校验并解码令牌，类型不符/过期/签名错误都抛 401。"""
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效令牌")

    if payload.get("type") != expected_type:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌类型错误")
    return payload


# ── FastAPI 依赖：从请求头解析当前患者 ──
async def get_current_patient(token: str = Depends(oauth2_scheme)) -> PatientProfile:
    """校验访问令牌并从数据库加载当前患者。"""
    payload = decode_token(token, expected_type="access")
    patient_id = payload.get("sub")
    if not patient_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌缺少主体")

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PatientProfile).where(PatientProfile.patient_id == patient_id)
        )
        patient = result.scalar_one_or_none()

    if patient is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
    return patient


async def get_current_doctor(token: str = Depends(oauth2_scheme)) -> PatientProfile:
    """校验当前用户为医生角色（人工介入接口受限）。"""
    patient = await get_current_patient(token)
    if patient.role != "doctor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="仅医生可访问")
    return patient