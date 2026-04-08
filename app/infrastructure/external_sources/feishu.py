from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from app.infrastructure.external_sources.base import ExternalSourceItem, ExternalSourcePage


@dataclass(frozen=True)
class FeishuSourceReference:
    source_kind: str
    token: str
    title_hint: str | None = None


@dataclass(frozen=True)
class FeishuDocumentContent:
    title: str
    content: str
    source_type: str
    source_uri: str
    connector: str = "feishu"
    external_document_id: str = ""
    external_version: str | None = None


class FeishuClient:
    """Fetches Feishu document content through official tenant-token and document APIs."""

    provider = "feishu"

    def __init__(
        self,
        base_url: str,
        app_id: str | None,
        app_secret: str | None,
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.app_id = app_id
        self.app_secret = app_secret
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def can_execute(self) -> bool:
        """Return whether enough Feishu credentials are configured for real API calls."""

        return bool(self.app_id and self.app_secret)

    def parse_source(self, source: str) -> FeishuSourceReference:
        """Extract a Feishu document or wiki token from a URL or raw token string."""

        if source.startswith("http://") or source.startswith("https://"):
            parsed = urlparse(source)
            parts = [part for part in parsed.path.split("/") if part]
            if "wiki" in parts:
                token = parts[-1]
                return FeishuSourceReference(source_kind="wiki", token=token)
            if "docx" in parts:
                token = parts[-1]
                return FeishuSourceReference(source_kind="docx", token=token)
            if parsed.query:
                query = parse_qs(parsed.query)
                if "token" in query:
                    return FeishuSourceReference(source_kind="wiki", token=query["token"][0])
            raise ValueError("Unsupported Feishu source URL.")

        compact = source.strip()
        if compact.startswith("wiki_"):
            return FeishuSourceReference(source_kind="wiki", token=compact)
        return FeishuSourceReference(source_kind="docx", token=compact)

    def fetch_document(self, source: str) -> FeishuDocumentContent:
        """Resolve one Feishu URL or token into normalized title and markdown-like raw content."""

        reference = self.parse_source(source)
        token = self.get_tenant_access_token()

        if reference.source_kind == "wiki":
            resolved = self.get_node(reference.token, token)
            if resolved["obj_type"] != "docx":
                raise ValueError("Current Feishu integration only supports wiki nodes backed by docx documents.")
            document = self.fetch_docx_document(resolved["obj_token"], token)
            title = resolved.get("title") or document["title"]
            return FeishuDocumentContent(
                title=title,
                content=document["content"],
                source_type="markdown",
                source_uri=source,
                external_document_id=resolved["obj_token"],
            )

        document = self.fetch_docx_document(reference.token, token)
        return FeishuDocumentContent(
            title=document["title"],
            content=document["content"],
            source_type="markdown",
            source_uri=source,
            external_document_id=reference.token,
        )

    def list_sources(
        self,
        cursor: str | None = None,
        limit: int = 20,
        source_root: str | None = None,
        space_id: str | None = None,
        parent_node_token: str | None = None,
    ) -> ExternalSourcePage:
        """List Feishu spaces or wiki child nodes with cursor-based pagination."""

        token = self.get_tenant_access_token()
        page_size = max(1, min(int(limit), 200))
        page_token = cursor or None

        if source_root:
            scope = self._resolve_listing_scope(source_root, token)
            space_id = scope["space_id"]
            parent_node_token = scope["parent_node_token"]

        if space_id:
            return self._list_space_nodes(
                access_token=token,
                space_id=space_id,
                page_token=page_token,
                page_size=page_size,
                parent_node_token=parent_node_token,
            )

        return self._list_spaces(
            access_token=token,
            page_token=page_token,
            page_size=page_size,
        )

    def health_check(self) -> dict:
        """Return whether Feishu credentials are configured and tenant token retrieval succeeds."""

        detail = {"base_url": self.base_url}
        reachable = False
        if self.can_execute():
            try:
                self.get_tenant_access_token()
                reachable = True
            except Exception as exc:
                detail["error"] = str(exc)
        return {
            "backend": self.provider,
            "execute_enabled": self.can_execute(),
            "reachable": reachable,
            "detail": detail,
        }

    def get_tenant_access_token(self) -> str:
        """Exchange app credentials for a tenant access token."""

        if not self.can_execute():
            raise RuntimeError("Feishu client is not configured")

        payload = self._request(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            json={
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            },
        )
        token = payload.get("tenant_access_token")
        if not token:
            raise RuntimeError("Feishu tenant token response did not include tenant_access_token")
        return str(token)

    def get_node(self, token: str, access_token: str) -> dict[str, str]:
        """Resolve one wiki node token into its backing object token and type."""

        payload = self._request(
            "GET",
            "/wiki/v2/spaces/get_node",
            access_token=access_token,
            params={"token": token},
        )
        data = payload.get("data", {})
        node = data.get("node") if isinstance(data, dict) else None
        if not isinstance(node, dict):
            raise RuntimeError("Feishu wiki node response did not include node data")
        obj_token = node.get("obj_token")
        obj_type = node.get("obj_type")
        if not obj_token or not obj_type:
            raise RuntimeError("Feishu wiki node response is missing obj_token or obj_type")
        return {
            "space_id": str(node.get("space_id") or ""),
            "node_token": str(node.get("node_token") or token),
            "obj_token": str(obj_token),
            "obj_type": str(obj_type),
            "parent_node_token": str(node.get("parent_node_token") or ""),
            "title": str(node.get("title") or ""),
        }

    def resolve_wiki_node(self, token: str, access_token: str) -> dict[str, str]:
        """Backward-compatible alias for one wiki node lookup."""

        return self.get_node(token, access_token)

    def fetch_docx_document(self, token: str, access_token: str) -> dict[str, str]:
        """Fetch raw content for one Feishu docx document."""

        payload = self._request(
            "GET",
            f"/docx/v1/documents/{token}/raw_content",
            access_token=access_token,
        )
        data = payload.get("data", {})
        content = data.get("content")
        title = data.get("title")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Feishu docx response did not include raw content")
        return {
            "title": str(title or token),
            "content": content.strip(),
        }

    def _resolve_listing_scope(self, source_root: str, access_token: str) -> dict[str, str]:
        reference = self.parse_source(source_root)
        if reference.source_kind != "wiki":
            raise ValueError("Feishu listing currently requires a wiki root URL or wiki token.")
        node = self.get_node(reference.token, access_token)
        if not node.get("space_id"):
            raise RuntimeError("Feishu node response did not include space_id.")
        return {
            "space_id": node["space_id"],
            "parent_node_token": node.get("node_token") or reference.token,
        }

    def _list_spaces(
        self,
        access_token: str,
        page_token: str | None,
        page_size: int,
    ) -> ExternalSourcePage:
        payload = self._request(
            "GET",
            "/wiki/v2/spaces",
            access_token=access_token,
            params={
                "page_size": str(page_size),
                **({"page_token": page_token} if page_token else {}),
            },
        )
        data = payload.get("data", {})
        items = data.get("items") if isinstance(data, dict) else []
        page_items: list[ExternalSourceItem] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            current_space_id = str(item.get("space_id") or "")
            if not current_space_id:
                continue
            page_items.append(
                ExternalSourceItem(
                    source=f"feishu://space/{current_space_id}",
                    source_kind="space",
                    external_document_id=current_space_id,
                    title=str(item.get("name") or current_space_id),
                    space_id=current_space_id,
                    has_child=True,
                )
            )
        return ExternalSourcePage(
            items=page_items,
            next_cursor=str(data.get("page_token") or "") if data.get("has_more") else None,
        )

    def _list_space_nodes(
        self,
        access_token: str,
        space_id: str,
        page_token: str | None,
        page_size: int,
        parent_node_token: str | None,
    ) -> ExternalSourcePage:
        params: dict[str, str] = {"page_size": str(page_size)}
        if page_token:
            params["page_token"] = page_token
        if parent_node_token:
            params["parent_node_token"] = parent_node_token
        payload = self._request(
            "GET",
            f"/wiki/v2/spaces/{space_id}/nodes",
            access_token=access_token,
            params=params,
        )
        data = payload.get("data", {})
        items = data.get("items") if isinstance(data, dict) else []
        page_items: list[ExternalSourceItem] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            node_token = str(item.get("node_token") or "")
            if not node_token:
                continue
            page_items.append(
                ExternalSourceItem(
                    source=f"https://feishu.cn/wiki/{node_token}",
                    source_kind="wiki",
                    external_document_id=str(item.get("obj_token") or node_token),
                    title=str(item.get("title") or node_token),
                    space_id=str(item.get("space_id") or space_id),
                    node_token=node_token,
                    parent_node_token=str(item.get("parent_node_token") or ""),
                    obj_type=str(item.get("obj_type") or ""),
                    has_child=bool(item.get("has_child")),
                )
            )
        return ExternalSourcePage(
            items=page_items,
            next_cursor=str(data.get("page_token") or "") if data.get("has_more") else None,
        )

    def _request(
        self,
        method: str,
        path: str,
        access_token: str | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        client = self._http_client or httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds)
        close_client = self._http_client is None
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        try:
            response = client.request(method, path, headers=headers, params=params, json=json)
            response.raise_for_status()
            payload = response.json()
        finally:
            if close_client:
                client.close()

        if payload.get("code", 0) not in {0, "0", None}:
            raise RuntimeError(str(payload.get("msg") or payload))
        return payload
