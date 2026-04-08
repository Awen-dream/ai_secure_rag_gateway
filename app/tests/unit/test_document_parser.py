import io
import unittest
import zipfile
from unittest.mock import patch

from app.application.ingestion.document_parser import (
    extract_text_from_bytes,
    infer_source_type,
    normalize_text_content,
)


def build_docx_bytes(*paragraphs: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        document_xml = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>',
        ]
        for paragraph in paragraphs:
            document_xml.append(f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>")
        document_xml.append("</w:body></w:document>")
        archive.writestr("word/document.xml", "".join(document_xml))
    return buffer.getvalue()


class _FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    def __init__(self, _stream) -> None:
        self.pages = [_FakePdfPage("第一页"), _FakePdfPage("第二页")]


class DocumentParserTest(unittest.TestCase):
    def test_infer_source_type_from_extension(self) -> None:
        self.assertEqual(infer_source_type(file_name="a.pdf"), "pdf")
        self.assertEqual(infer_source_type(file_name="a.docx"), "docx")
        self.assertEqual(infer_source_type(file_name="a.md"), "markdown")
        self.assertEqual(infer_source_type(file_name="a.html"), "html")

    def test_normalize_html_content_strips_tags_and_keeps_blocks(self) -> None:
        text = normalize_text_content("<h1>制度</h1><p>审批时限为3个工作日。</p>", "html")

        self.assertIn("制度", text)
        self.assertIn("审批时限为3个工作日。", text)
        self.assertNotIn("<h1>", text)

    def test_extract_docx_text_reads_word_document_xml(self) -> None:
        content = extract_text_from_bytes(build_docx_bytes("采购流程", "审批时限为2个工作日。"), "docx", "demo.docx")

        self.assertIn("采购流程", content)
        self.assertIn("审批时限为2个工作日。", content)

    def test_extract_pdf_text_uses_pypdf_reader(self) -> None:
        with patch("app.application.ingestion.document_parser.PdfReader", _FakePdfReader):
            content = extract_text_from_bytes(b"%PDF-mock", "pdf", "demo.pdf")

        self.assertEqual(content, "第一页\n第二页")


if __name__ == "__main__":
    unittest.main()
