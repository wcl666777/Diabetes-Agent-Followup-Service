"""患者个人文档：上传落库 → 父子分块 → 本地检索（BM25 命中 child → 回溯 parent）。

数据流：
  上传（纯文本）→ PatientDocument（全文） + PatientDocChunk（parent/child 两块）
  检索：对 child 块做 BM25 关键词打分，命中后返回对应 parent 块作为上下文。

只存纯文本，不保留原文件、不做向量化（与「存文本即可」的约定一致）。
"""
from typing import List

from sqlalchemy import delete, select

from app.core.chunking import build_parent_child_chunks
from app.core.db import get_session_factory
from app.core.loading import parse_bytes
from app.core.logger import get_logger
from app.core.models import PatientDocChunk, PatientDocument

logger = get_logger(__name__)

# 合法文档类型：病例 / 出院小结，分开存储
DOC_TYPES = {"case_record": "病例", "discharge_summary": "出院小结"}


def _tokenize(text: str) -> List[str]:
    """中文分词（jieba），用于 BM25 关键词检索。"""
    import jieba

    return [w for w in jieba.cut(text) if w.strip()]


async def create_document(patient_id: str, doc_type: str, content: str, title: str = "") -> dict:
    """解析后的纯文本写入文档表 + 父子分块表。content 为纯文本全文。"""
    if doc_type not in DOC_TYPES:
        raise ValueError(f"非法文档类型：{doc_type}")

    content = content.strip()
    if not content:
        raise ValueError("文档内容为空")

    chunks = build_parent_child_chunks(content)
    if not chunks:
        raise ValueError("文档内容无法分块")

    factory = get_session_factory()
    async with factory() as session:
        doc = PatientDocument(
            patient_id=patient_id,
            doc_type=doc_type,
            title=title or DOC_TYPES[doc_type],
            content=content,
            char_count=len(content),
        )
        session.add(doc)
        await session.flush()  # 拿到自增 id

        for c in chunks:
            session.add(
                PatientDocChunk(
                    document_id=doc.id,
                    patient_id=patient_id,
                    doc_type=doc_type,
                    parent_index=c["parent_index"],
                    level=c["level"],
                    content=c["content"],
                )
            )
        await session.commit()

    logger.info(
        "patient_docs.created | patient=%s | type=%s | chars=%d | chunks=%d",
        patient_id, doc_type, len(content), len(chunks),
    )
    return {
        "id": doc.id,
        "doc_type": doc_type,
        "doc_type_label": DOC_TYPES[doc_type],
        "title": doc.title,
        "char_count": len(content),
        "chunk_count": len(chunks),
    }


async def list_documents(patient_id: str) -> List[dict]:
    """列出当前患者已上传的文档（按时间倒序）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PatientDocument)
            .where(PatientDocument.patient_id == patient_id)
            .order_by(PatientDocument.created_at.desc())
        )
        rows = result.scalars().all()

    return [
        {
            "id": d.id,
            "doc_type": d.doc_type,
            "doc_type_label": DOC_TYPES.get(d.doc_type, d.doc_type),
            "title": d.title,
            "char_count": d.char_count,
            "created_at": d.created_at.isoformat() if d.created_at else "",
        }
        for d in rows
    ]


async def delete_document(patient_id: str, doc_id: int) -> bool:
    """删除某份文档及其分块（级联清理，不残留孤儿 chunk）。"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(PatientDocument).where(
                PatientDocument.id == doc_id,
                PatientDocument.patient_id == patient_id,
            )
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            return False

        await session.execute(
            delete(PatientDocChunk).where(PatientDocChunk.document_id == doc_id)
        )
        await session.delete(doc)
        await session.commit()
    logger.info("patient_docs.deleted | patient=%s | doc_id=%s", patient_id, doc_id)
    return True


async def retrieve_personal(patient_id: str, question: str, top_k: int = 3) -> List[dict]:
    """个人文档检索：child 块做 BM25 打分命中，返回其 parent 块作为完整上下文。

    病例与出院小结两类文档一起参与检索（满足「结合两类检索」的需求）。
    返回 [{"doc_type", "doc_type_label", "title", "content"}]，content 为 parent 块文本。
    """
    factory = get_session_factory()

    # ① 取该患者全部 child 块（个人文档量小，全量加载构建 BM25）
    async with factory() as session:
        result = await session.execute(
            select(PatientDocChunk).where(
                PatientDocChunk.patient_id == patient_id,
                PatientDocChunk.level == "child",
            )
        )
        children = result.scalars().all()

    if not children:
        return []

    from rank_bm25 import BM25Okapi

    q_tokens = _tokenize(question)
    qset = set(q_tokens)
    corpus = [_tokenize(c.content) for c in children]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q_tokens)

    # ② 命中 child：与问题至少共享一个词才视为命中。BM25 得分仅用于排序；
    #    注意：小语料下 BM25 得分可能为负（IDF 为负），负值只代表词在语料中较常见，
    #    不影响命中判断，故用「共享词」而不是分数正负来判定。
    hit_idx = [i for i, doc in enumerate(corpus) if qset & set(doc)]
    if not hit_idx:
        return []
    hit_idx.sort(key=lambda i: scores[i], reverse=True)
    hit_children = [children[i] for i in hit_idx[:top_k]]

    # ③ 回溯 parent：按 (document_id, parent_index) 去重，命中多个 child 只返回一次 parent
    parents_needed = {(c.document_id, c.parent_index) for c in hit_children}

    async with factory() as session:
        parent_result = await session.execute(
            select(PatientDocChunk).where(
                PatientDocChunk.patient_id == patient_id,
                PatientDocChunk.level == "parent",
            )
        )
        all_parents = parent_result.scalars().all()
        doc_result = await session.execute(
            select(PatientDocument).where(PatientDocument.patient_id == patient_id)
        )
        docs = {d.id: d for d in doc_result.scalars().all()}

    parent_map = {(p.document_id, p.parent_index): p for p in all_parents}

    out: List[dict] = []
    for (doc_id, parent_idx) in parents_needed:
        parent = parent_map.get((doc_id, parent_idx))
        if parent is None:
            continue
        doc = docs.get(doc_id)
        out.append(
            {
                "doc_type": parent.doc_type,
                "doc_type_label": DOC_TYPES.get(parent.doc_type, parent.doc_type),
                "title": doc.title if doc else "",
                "content": parent.content,
            }
        )

    logger.info(
        "patient_docs.retrieved | patient=%s | question=%s | parents=%d",
        patient_id, question[:30], len(out),
    )
    return out