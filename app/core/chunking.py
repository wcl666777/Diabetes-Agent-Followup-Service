"""中文感知的父子块分块。

父子块思路：
- parent 块（较大，~1000 字符）：作为完整上下文送给大模型，不直接参与检索命中
- child 块（较小，~300 字符）：参与检索命中，命中后回溯其所属 parent 块

与 RAG 服务的 ingestion/splitter.py 使用同一套中文 separators，保证两个服务的
分块口径一致。
"""
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

_ZH_SEPARATORS = [
    "\n\n",     # 段落
    "\n",       # 换行
    "。", "！", "？", "；", "，",  # 中文句末/停顿标点
    ".", "!", "?", ";",  # 英文标点
    " ",        # 空格
    "",         # 兜底：按字符硬切
]

# 分块参数（个人文档粒度，如调整需与检索联动）
PARENT_SIZE = 1000
PARENT_OVERLAP = 150
CHILD_SIZE = 300
CHILD_OVERLAP = 50


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_ZH_SEPARATORS,
        length_function=len,
    )


def build_parent_child_chunks(
    text: str,
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> List[dict]:
    """把全文切分为父子块列表。

    先切 parent（大块），再把每个 parent 切成多个 child，
    child 用 parent_index 指向所属 parent，保证「命中 child → 回溯 parent」一一对应。

    返回 [{"parent_index": int, "level": "parent"|"child", "content": str}]。
    """
    parent_splitter = _make_splitter(parent_size, parent_overlap)
    child_splitter = _make_splitter(child_size, child_overlap)

    chunks: List[dict] = []
    parent_idx = 0
    for parent_piece in parent_splitter.split_text(text):
        parent = parent_piece.strip()
        if not parent:
            continue
        chunks.append({"parent_index": parent_idx, "level": "parent", "content": parent})
        for child_piece in child_splitter.split_text(parent):
            child = child_piece.strip()
            if child:
                chunks.append({"parent_index": parent_idx, "level": "child", "content": child})
        parent_idx += 1
    return chunks