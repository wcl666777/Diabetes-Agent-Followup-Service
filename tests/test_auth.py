"""认证模块（auth.py）单元测试：密码哈希 + JWT 签发/校验。

纯单元测试，不依赖数据库。运行：pytest tests/test_auth.py
"""
import pytest

from app.core.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswordHash:
    def test_hash_and_verify(self):
        h = hash_password("secret123")
        assert h != "secret123"
        assert verify_password("secret123", h)

    def test_wrong_password(self):
        h = hash_password("secret123")
        assert not verify_password("wrong", h)

    def test_hash_is_salted(self):
        # 同一明文两次哈希结果不同（bcrypt 随机盐）
        assert hash_password("same") != hash_password("same")


class TestJwt:
    def test_access_token_roundtrip(self):
        token = create_access_token("patient_001", "patient")
        payload = decode_token(token, expected_type="access")
        assert payload["sub"] == "patient_001"
        assert payload["role"] == "patient"

    def test_refresh_token_type(self):
        token = create_refresh_token("patient_001", "patient")
        payload = decode_token(token, expected_type="refresh")
        assert payload["type"] == "refresh"

    def test_wrong_type_rejected(self):
        token = create_access_token("patient_001", "patient")
        with pytest.raises(Exception):
            decode_token(token, expected_type="refresh")

    def test_invalid_token_rejected(self):
        with pytest.raises(Exception):
            decode_token("not.a.valid.token", expected_type="access")