from __future__ import annotations

import html
import io
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - import availability is verified in runtime and dependency setup.
    PdfReader = None


WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SUPPORTED_FILE_SOURCE_TYPES = {"pdf", "docx", "markdown", "html", "text", "manual"}


class _HTMLToTextParser(HTMLParser):
    """Converts HTML into readable plain text while preserving block boundaries."""

    block_tags = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "table",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.block_tags and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.block_tags and (not self.parts or not self.parts[-1].endswith("\n")):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = html.unescape(data)
        if text.strip():
            self.parts.append(text)

    def get_text(self) -> str:
        collapsed = "".join(self.parts)
        lines = [line.strip() for line in collapsed.splitlines()]
        return "\n".join(line for line in lines if line)


def infer_source_type(file_name: str | None = None, declared_type: str | None = None) -> str:
    """Infer one supported source type from explicit type or file extension."""

    if declared_type:
        normalized = declared_type.strip().lower()
        if normalized in {"md", "markdown"}:
            return "markdown"
        if normalized in {"htm", "html"}:
            return "html"
        if normalized in {"txt", "text", "manual"}:
            return "manual"
        if normalized in SUPPORTED_FILE_SOURCE_TYPES:
            return normalized

    suffix = Path(file_name or "").suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".txt", ".text"}:
        return "manual"
    return "manual"


def normalize_text_content(content: str, source_type: str) -> str:
    """Normalize text content for text-based uploads before chunking and indexing."""

    normalized_type = infer_source_type(declared_type=source_type)
    if normalized_type == "html":
        parser = _HTMLToTextParser()
        parser.feed(content)
        return parser.get_text().strip()
    return content.strip()


def extract_text_from_bytes(file_bytes: bytes, source_type: str, file_name: str | None = None) -> str:
    """Extract plain text from uploaded binary or text files based on source type."""

    normalized_type = infer_source_type(file_name=file_name, declared_type=source_type)
    if normalized_type == "pdf":
        return _extract_pdf_text(file_bytes)
    if normalized_type == "docx":
        return _extract_docx_text(file_bytes)
    if normalized_type == "html":
        return normalize_text_content(file_bytes.decode("utf-8"), "html")
    if normalized_type == "markdown":
        return file_bytes.decode("utf-8").strip()
    return file_bytes.decode("utf-8").strip()


def _extract_pdf_text(file_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is required to parse PDF documents.")

    reader = PdfReader(io.BytesIO(file_bytes))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(item.strip() for item in page_texts if item and item.strip()).strip()
    if not text:
        raise ValueError("PDF parser could not extract text from the uploaded file.")
    return text


def _extract_docx_text(file_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    lines: list[str] = []
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        pieces: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.split("}", 1)[-1]
            if tag == "t" and node.text:
                pieces.append(node.text)
            elif tag == "tab":
                pieces.append("\t")
            elif tag in {"br", "cr"}:
                pieces.append("\n")
        paragraph_text = "".join(pieces).strip()
        if paragraph_text:
            lines.append(paragraph_text)

    text = "\n".join(lines).strip()
    if not text:
        raise ValueError("DOCX parser could not extract text from the uploaded file.")
    return text
