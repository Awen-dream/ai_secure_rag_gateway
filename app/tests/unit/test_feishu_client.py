import unittest

import httpx

from app.infrastructure.external_sources.feishu import FeishuClient


class FeishuClientTest(unittest.TestCase):
    def test_parse_source_supports_docx_and_wiki_urls(self) -> None:
        client = FeishuClient(
            base_url="https://open.feishu.cn/open-apis",
            app_id="app_id",
            app_secret="app_secret",
        )

        doc = client.parse_source("https://example.feishu.cn/docx/AbCdEfGh")
        wiki = client.parse_source("https://example.feishu.cn/wiki/WikiToken")

        self.assertEqual(doc.source_kind, "docx")
        self.assertEqual(doc.token, "AbCdEfGh")
        self.assertEqual(wiki.source_kind, "wiki")
        self.assertEqual(wiki.token, "WikiToken")

    def test_fetch_docx_document_uses_tenant_token_and_raw_content_api(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/auth/v3/tenant_access_token/internal"):
                return httpx.Response(200, json={"code": 0, "tenant_access_token": "tenant-token"})
            if request.url.path.endswith("/docx/v1/documents/DocToken/raw_content"):
                self.assertEqual(request.headers["Authorization"], "Bearer tenant-token")
                return httpx.Response(
                    200,
                    json={"code": 0, "data": {"title": "报销制度", "content": "# 报销制度\n\n审批时限为3个工作日。"}},
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        client = FeishuClient(
            base_url="https://open.feishu.cn/open-apis",
            app_id="app_id",
            app_secret="app_secret",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://open.feishu.cn/open-apis"),
        )

        document = client.fetch_document("https://example.feishu.cn/docx/DocToken")

        self.assertEqual(document.title, "报销制度")
        self.assertIn("审批时限为3个工作日", document.content)
        self.assertEqual(document.source_type, "markdown")

    def test_fetch_wiki_document_resolves_node_before_loading_docx(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/auth/v3/tenant_access_token/internal"):
                return httpx.Response(200, json={"code": 0, "tenant_access_token": "tenant-token"})
            if request.url.path.endswith("/wiki/v2/spaces/get_node"):
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "data": {
                            "node": {
                                "title": "采购流程",
                                "obj_type": "docx",
                                "obj_token": "ResolvedDocToken",
                            }
                        },
                    },
                )
            if request.url.path.endswith("/docx/v1/documents/ResolvedDocToken/raw_content"):
                return httpx.Response(
                    200,
                    json={"code": 0, "data": {"title": "采购流程", "content": "# 采购流程\n\n审批时限为2个工作日。"}},
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        client = FeishuClient(
            base_url="https://open.feishu.cn/open-apis",
            app_id="app_id",
            app_secret="app_secret",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://open.feishu.cn/open-apis"),
        )

        document = client.fetch_document("https://example.feishu.cn/wiki/WikiToken")

        self.assertEqual(document.title, "采购流程")
        self.assertIn("审批时限为2个工作日", document.content)

    def test_list_sources_is_explicitly_not_configured_yet(self) -> None:
        client = FeishuClient(
            base_url="https://open.feishu.cn/open-apis",
            app_id="app_id",
            app_secret="app_secret",
        )

        with self.assertRaises(RuntimeError):
            client.list_sources(cursor=None, limit=20)


if __name__ == "__main__":
    unittest.main()
