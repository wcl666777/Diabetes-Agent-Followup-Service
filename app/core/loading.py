"""文档加载：把上传的 PDF / Word / Markdown / TXT 解析为纯文本。

与 RAG 服务的 ingestion/loader.py 保持一致的解析能力，但这里直接接收文件字节
（UploadFile），只产出纯文本，不落盘、不保留原文件。
"""
import io

from app.core.logger import get_logger

logger = get_logger(__name__)

# 支持的扩展名（不含点）
_ALLOWED_EXT = {"txt", "md", "markdown", "docx", "pdf"}


def parse_bytes(filename: str, data: bytes) -> str:
    """按文件名扩展名，把文件字节解析为纯文本字符串。"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXT:
        raise ValueError(f"不支持的文件类型：.{ext}（支持 txt / md / docx / pdf）")

    if ext in ("txt", "md", "markdown"):
        return _decode_text(data)
    if ext == "docx":
        return _parse_docx(data)
    if ext == "pdf":
        return _parse_pdf(data)
    return ""


def _decode_text(data: bytes) -> str:
    """解码文本文件，优先 UTF-8，失败回退 GB18030（兼容中文 Windows 导出）。"""
    for enc in ("utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _parse_docx(data: bytes) -> str:
    """解析 Word 文档，按段落拼接为纯文本。"""
    import docx

    d = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def _parse_pdf(data: bytes) -> str:
    """解析 PDF，逐页提取文本后用换行拼接。"""
    import pdfplumber

    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)