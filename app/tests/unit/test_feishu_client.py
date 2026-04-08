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

    def test_list_sources_can_list_spaces(self) -> None:
        requests_seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(str(request.url))
            if request.url.path.endswith("/auth/v3/tenant_access_token/internal"):
                return httpx.Response(200, json={"code": 0, "tenant_access_token": "tenant-token"})
            if request.url.path.endswith("/wiki/v2/spaces"):
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "data": {
                            "items": [{"space_id": "space_1", "name": "知识空间 1"}],
                            "page_token": "cursor_2",
                            "has_more": True,
                        },
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        client = FeishuClient(
            base_url="https://open.feishu.cn/open-apis",
            app_id="app_id",
            app_secret="app_secret",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://open.feishu.cn/open-apis"),
        )

        page = client.list_sources(cursor=None, limit=20)

        self.assertEqual(len(page.items), 1)
        self.assertEqual(page.items[0].source_kind, "space")
        self.assertEqual(page.items[0].space_id, "space_1")
        self.assertEqual(page.next_cursor, "cursor_2")
        self.assertTrue(any("/wiki/v2/spaces" in item for item in requests_seen))

    def test_list_sources_can_list_child_nodes_from_wiki_root(self) -> None:
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
                                "space_id": "space_1",
                                "node_token": "root_node",
                                "obj_token": "docx_root",
                                "obj_type": "docx",
                                "title": "根节点",
                            }
                        },
                    },
                )
            if request.url.path.endswith("/wiki/v2/spaces/space_1/nodes"):
                self.assertEqual(request.url.params.get("parent_node_token"), "root_node")
                self.assertEqual(request.url.params.get("page_size"), "2")
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "data": {
                            "items": [
                                {
                                    "space_id": "space_1",
                                    "node_token": "child_node_1",
                                    "obj_type": "docx",
                                    "parent_node_token": "root_node",
                                    "has_child": False,
                                    "title": "子文档 1",
                                }
                            ],
                            "page_token": "cursor_2",
                            "has_more": True,
                        },
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        client = FeishuClient(
            base_url="https://open.feishu.cn/open-apis",
            app_id="app_id",
            app_secret="app_secret",
            http_client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://open.feishu.cn/open-apis"),
        )

        page = client.list_sources(source_root="https://example.feishu.cn/wiki/root_node", limit=2)

        self.assertEqual(len(page.items), 1)
        self.assertEqual(page.items[0].source, "https://feishu.cn/wiki/child_node_1")
        self.assertEqual(page.items[0].source_kind, "wiki")
        self.assertEqual(page.items[0].parent_node_token, "root_node")
        self.assertEqual(page.next_cursor, "cursor_2")


if __name__ == "__main__":
    unittest.main()
