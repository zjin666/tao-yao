"""把收到的文件读成一小段文字：文本类、PDF、Word、Excel、PowerPoint。

只在本机解析，不上传。超过长度上限的只读开头并说明；读不了的（加密、扫描件、不支持的格式）如实说明，不编造内容。
"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

log = logging.getLogger("taoyao.filetext")

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".log", ".ini", ".cfg", ".conf", ".toml",
    ".yaml", ".yml", ".xml", ".html", ".htm", ".srt", ".tex", ".sql", ".py", ".js", ".ts", ".java", ".c", ".h",
    ".cpp", ".cs", ".go", ".rs", ".sh", ".bat", ".ps1", ".css", ".vue", ".kt", ".swift", ".rb", ".php",
}
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


@dataclass
class FileResult:
    status: str          # ok / unsupported / empty / too_large / error
    kind: str = ""       # 文本 / PDF / Word / Excel / PowerPoint
    text: str = ""
    truncated: bool = False
    detail: str = ""


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _clip(text: str, limit: int) -> tuple[str, bool]:
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    return (text[:limit], True) if len(text) > limit else (text, False)


def _pdf(path: Path, max_pages: int) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", "没有装 PDF 读取库"
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        return "", "PDF 有密码，打不开"
    parts = []
    for index, page in enumerate(reader.pages):
        if index >= max_pages:
            parts.append(f"……（共 {len(reader.pages)} 页，只读了前 {max_pages} 页）")
            break
        parts.append(page.extract_text() or "")
    return "\n".join(parts), ""


def _docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.iter(f"{W_NS}p"):
        paragraphs.append("".join(node.text or "" for node in paragraph.iter(f"{W_NS}t")))
    return "\n".join(paragraphs)


def _xlsx(path: Path, max_rows: int = 60) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    for sheet in workbook.worksheets[:5]:
        parts.append(f"【工作表：{sheet.title}】")
        for index, row in enumerate(sheet.iter_rows(values_only=True)):
            if index >= max_rows:
                parts.append("……（行数较多，只读了前面一部分）")
                break
            cells = ["" if value is None else str(value) for value in row]
            if any(cells):
                parts.append("\t".join(cells).rstrip())
    workbook.close()
    return "\n".join(parts)


def _pptx(path: Path) -> str:
    parts = []
    with zipfile.ZipFile(path) as archive:
        names = sorted((n for n in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                       key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
        for number, name in enumerate(names, 1):
            root = ElementTree.fromstring(archive.read(name))
            text = " ".join(node.text or "" for node in root.iter(f"{A_NS}t")).strip()
            if text:
                parts.append(f"【第 {number} 页】{text}")
    return "\n".join(parts)


def extract_text(path: Path, name: str = "", max_chars: int = 4000, max_bytes: int = 10 * 1024 * 1024,
                 max_pages: int = 30) -> FileResult:
    path = Path(path)
    suffix = Path(name or path.name).suffix.lower()
    try:
        if path.stat().st_size > max_bytes:
            return FileResult("too_large", detail=f"文件超过 {max_bytes // 1048576} MB，没有读")
        if suffix in TEXT_EXTENSIONS:
            kind, raw = "文本", _decode(path.read_bytes()[:2 * 1024 * 1024])
        elif suffix == ".pdf":
            kind = "PDF"
            raw, problem = _pdf(path, max_pages)
            if problem:
                return FileResult("error", kind, detail=problem)
        elif suffix == ".docx":
            kind, raw = "Word", _docx(path)
        elif suffix in (".xlsx", ".xlsm"):
            kind, raw = "Excel", _xlsx(path)
        elif suffix == ".pptx":
            kind, raw = "PowerPoint", _pptx(path)
        else:
            return FileResult("unsupported", detail=f"暂时读不了 {suffix or '这种'} 格式的文件")
    except Exception as exc:
        log.info("读取文件失败 %s：%s", path.name, exc)
        return FileResult("error", detail="文件打不开或已损坏")
    text, truncated = _clip(raw, max_chars)
    if not text:
        return FileResult("empty", kind, detail="没读到文字（可能是扫描件或图片）")
    return FileResult("ok", kind, text, truncated)
