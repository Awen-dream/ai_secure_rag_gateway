import unittest

from app.application.ingestion.pipelines import chunk_document, chunk_text, estimate_token_count


class ChunkTextTests(unittest.TestCase):
    def test_chunk_text_splits_chinese_content_by_token_budget(self) -> None:
        content = (
            "这是第一段内容，主要介绍系统背景和目标，适合作为RAG检索语料。"
            "\n\n"
            "这是第二段内容，补充说明安全约束、权限边界和数据处理要求。"
            "\n\n"
            "这是第三段内容，用来验证中文文本在没有空格时也能被稳定切分。"
        )

        chunks = chunk_text(content, max_tokens=28, overlap_tokens=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(estimate_token_count(chunk) <= 28 for chunk in chunks))

    def test_chunk_text_splits_long_paragraph_without_blank_lines(self) -> None:
        content = (
            "系统需要支持文档接入、清洗、切块、嵌入和检索。"
            "同时还要保证租户隔离、权限控制和审计可追踪。"
            "如果某一段特别长，也不能整段直接塞进一个chunk里。"
            "这里故意构造一个没有空行的长段落，验证会继续按句子回退切分。"
        )

        chunks = chunk_text(content, max_tokens=24, overlap_tokens=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(estimate_token_count(chunk) <= 24 for chunk in chunks))

    def test_chunk_text_keeps_overlap_between_chunks(self) -> None:
        paragraph_1 = "alpha beta gamma delta"
        paragraph_2 = "epsilon zeta eta theta"
        paragraph_3 = "iota kappa lambda mu"
        content = f"{paragraph_1}\n\n{paragraph_2}\n\n{paragraph_3}"

        chunks = chunk_text(content, max_tokens=10, overlap_tokens=4)

        self.assertEqual(len(chunks), 2)
        self.assertIn(paragraph_2, chunks[0])
        self.assertIn(paragraph_2, chunks[1])

    def test_chunk_document_prioritizes_headings_as_section_boundaries(self) -> None:
        content = (
            "# Access Control\n"
            "Tenant isolation is enforced before retrieval.\n\n"
            "## Least Privilege\n"
            "Users should only receive documents within their clearance.\n\n"
            "# Audit\n"
            "Every access decision must be traceable."
        )

        chunks = chunk_document(content, max_tokens=80, overlap_tokens=10)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].section_name, "Access Control")
        self.assertEqual(chunks[1].section_name, "Least Privilege")
        self.assertEqual(chunks[1].heading_path, ["Access Control", "Least Privilege"])
        self.assertEqual(chunks[2].section_name, "Audit")
        self.assertIn("Access Control", chunks[0].text)
        self.assertIn("Least Privilege", chunks[1].text)
        self.assertIn("Audit", chunks[2].text)


if __name__ == "__main__":
    unittest.main()
