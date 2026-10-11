"""患者个人文档接口：上传（文件/文本）→ 解析 → 父子分块落库；列表；删除。

只存储解析后的纯文本，不保留原文件；病例与出院小结按 doc_type 分开存储。
"""
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.auth import get_current_patient
from app.core.loading import parse_bytes
from app.core.logger import get_logger
from app.core.models import PatientProfile
from app.core.patient_docs import (
    DOC_TYPES,
    create_document,
    delete_document,
    list_documents,
)

router = APIRouter(prefix="/api/v1/documents", tags=["患者个人文档"])
logger = get_logger(__name__)

_MAX_FILE_BYTES = 10 * 1024 * 1024  # 单文件上限 10MB


@router.post("")
async def upload_document(
    doc_type: str = Form(...),
    title: str = Form(""),
    text: str = Form(""),
    file: Optional[UploadFile] = File(None),
    patient: PatientProfile = Depends(get_current_patient),
):
    """上传一份病例或出院小结。

    文件与文本二选一：有文件则解析文件，否则使用粘贴的文本。
    """
    if doc_type not in DOC_TYPES:
        raise HTTPException(status_code=400, detail=f"doc_type 仅支持 {list(DOC_TYPES)}")

    content = text.strip() if text else ""
    if file is not None:
        data = await file.read()
        if len(data) > _MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 10MB")
        try:
            content = parse_bytes(file.filename or "", data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    if not content.strip():
        raise HTTPException(status_code=400, detail="请上传文件或粘贴文本内容")

    try:
        return await create_document(patient.patient_id, doc_type, content, title)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
async def list_docs(patient: PatientProfile = Depends(get_current_patient)):
    """列出当前患者已上传的文档。"""
    return {"documents": await list_documents(patient.patient_id)}


@router.delete("/{doc_id}")
async def delete_doc(doc_id: int, patient: PatientProfile = Depends(get_current_patient)):
    """删除某份文档及其分块。"""
    ok = await delete_document(patient.patient_id, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在或无权删除")
    return {"status": "ok"}