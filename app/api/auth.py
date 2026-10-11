"""认证接口：注册 / 登录 / 刷新令牌。"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.core.db import get_session_factory
from app.core.logger import get_logger
from app.core.models import PatientProfile

router = APIRouter(prefix="/api/v1/auth", tags=["认证"])
logger = get_logger(__name__)


class RegisterRequest(BaseModel):
    patient_id: str = Field(..., min_length=3, max_length=64, description="患者 ID（登录账号）")
    password: str = Field(..., min_length=6, max_length=64, description="密码（至少 6 位）")
    name: str = Field("", max_length=64)
    role: str = Field("patient", description="patient / doctor")


class LoginRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=64)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    patient_id: str
    role: str


class RefreshRequest(BaseModel):
    refresh_token: str = Field(...)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """注册新患者/医生账号，返回一对令牌（注册即登录）。"""
    if req.role not in ("patient", "doctor"):
        raise HTTPException(status_code=400, detail="role 只能是 patient 或 doctor")

    factory = get_session_factory()
    async with factory() as session:
        existing = await session.execute(
            select(PatientProfile).where(PatientProfile.patient_id == req.patient_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="账号已存在")

        patient = PatientProfile(
            patient_id=req.patient_id,
            password_hash=hash_password(req.password),
            name=req.name,
            role=req.role,
        )
        session.add(patient)
        await session.commit()
        await session.refresh(patient)

    logger.info("auth.register | patient_id=%s role=%s", req.patient_id, req.role)
    return TokenResponse(
        access_token=create_access_token(patient.patient_id, patient.role),
        refresh_token=create_refresh_token(patient.patient_id, patient.role),
        patient_id=patient.patient_id,
        role=patient.role,
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """登录：校验账号密码，返回访问 + 刷新令牌。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PatientProfile).where(PatientProfile.patient_id == req.patient_id)
        )
        patient = result.scalar_one_or_none()

    if patient is None or not verify_password(req.password, patient.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号或密码错误")

    logger.info("auth.login | patient_id=%s", req.patient_id)
    return TokenResponse(
        access_token=create_access_token(patient.patient_id, patient.role),
        refresh_token=create_refresh_token(patient.patient_id, patient.role),
        patient_id=patient.patient_id,
        role=patient.role,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest):
    """用刷新令牌换取新访问令牌（无感续期）。"""
    payload = decode_token(req.refresh_token, expected_type="refresh")
    patient_id = payload.get("sub")

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PatientProfile).where(PatientProfile.patient_id == patient_id)
        )
        patient = result.scalar_one_or_none()

    if patient is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")

    return TokenResponse(
        access_token=create_access_token(patient.patient_id, patient.role),
        refresh_token=create_refresh_token(patient.patient_id, patient.role),
        patient_id=patient.patient_id,
        role=patient.role,
    )