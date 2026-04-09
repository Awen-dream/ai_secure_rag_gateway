from __future__ import annotations

import json
from urllib import error, request
from typing import Sequence

from app.application.access.service import AccessFilter
from app.domain.documents.models import DocumentChunk, DocumentRecord
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.models import RetrievalBackendHealth, RetrievalBackendInfo
from app.domain.retrieval.retrievers import keyword_features


class ElasticsearchSearch:
    """Development-safe Elasticsearch adapter.

    The public interface is intentionally aligned with the real backend contract we want
    to keep later. In local development it scores permission-filtered candidates
    in-process, which lets the hybrid retrieval pipeline stay runnable without an
    external Elasticsearch cluster.
    """

    backend_name = "elasticsearch"

    def __init__(
        self,
        index_name: str = "knowledge_chunks",
        mode: str = "local-fallback",
        endpoint: str | None = None,
        auto_init_index: bool = False,
    ) -> None:
        self.index_name = index_name
        self.mode = mode
        self.endpoint = endpoint
        self.auto_init_index = auto_init_index

    def search(
        self,
        query: str,
        terms: Sequence[str],
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
        access_filter: AccessFilter | None = None,
        tag_filters: Sequence[str] | None = None,
        year_filters: Sequence[int] | None = None,
        exact_terms: Sequence[str] | None = None,
    ) -> list[BackendSearchHit]:
        """Return keyword-oriented hits using title and body term matching."""

        if self.can_execute() and candidates:
            try:
                return self._remote_search(
                    query,
                    terms,
                    candidates,
                    top_k,
                    access_filter,
                    tag_filters=tag_filters,
                    year_filters=year_filters,
                    exact_terms=exact_terms,
                )
            except Exception:
                pass

        hits: list[BackendSearchHit] = []
        for document, chunk in candidates:
            score, matched_terms = keyword_features(list(terms), document, chunk)
            if score <= 0:
                continue
            hits.append(
                BackendSearchHit(
                    document=document,
                    chunk=chunk,
                    score=score,
                    backend=self.backend_name,
                    matched_terms=matched_terms,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]

    def upsert_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Sync or preview keyword index updates for the given document."""

        executed = False
        if self.can_execute() and chunks:
            try:
                if self.auto_init_index:
                    self.initialize_index()
                self._bulk_upsert(document, chunks)
                executed = True
            except Exception:
                executed = False

        return {
            "backend": self.backend_name,
            "index": self.index_name,
            "doc_id": document.id,
            "chunks_indexed": len(chunks),
            "bulk_preview_lines": len(self.build_bulk_payload(document, chunks).splitlines()),
            "executed": executed,
        }

    def delete_document(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Remove all indexed chunks for one document."""

        executed = False
        if self.can_execute() and chunks:
            try:
                self._bulk_delete(chunks)
                executed = True
            except Exception:
                executed = False

        return {
            "backend": self.backend_name,
            "index": self.index_name,
            "doc_id": document.id,
            "chunks_deleted": len(chunks),
            "executed": executed,
        }

    def describe_backend(self) -> RetrievalBackendInfo:
        """Return deployment and capability metadata for the keyword backend."""

        return RetrievalBackendInfo(
            backend=self.backend_name,
            mode=self.mode,
            config={
                "index_name": self.index_name,
                "endpoint": self.endpoint,
                "auto_init_index": self.auto_init_index,
            },
            capabilities=[
                "keyword_search",
                "bm25_style_matching",
                "metadata_filtering",
                "bulk_index_preview",
                "mapping_preview",
                "remote_search",
                "health_check",
            ],
        )

    def can_execute(self) -> bool:
        """Return whether this adapter is configured to talk to a real Elasticsearch cluster."""

        return self.mode == "remote" and bool(self.endpoint)

    def health_check(self) -> RetrievalBackendHealth:
        """Return reachability status for the configured Elasticsearch backend."""

        reachable = False
        detail = {"endpoint": self.endpoint, "index_name": self.index_name}
        if self.can_execute():
            try:
                payload = self._request_json("GET", "/")
                reachable = True
                detail["cluster"] = payload
            except Exception as exc:
                detail["error"] = str(exc)
        return RetrievalBackendHealth(
            backend=self.backend_name,
            execute_enabled=self.can_execute(),
            reachable=reachable,
            detail=detail,
        )

    def initialize_index(self) -> dict:
        """Create the Elasticsearch index with the expected mapping when execution is enabled."""

        mapping = self.build_index_mapping()
        executed = False
        response_payload: dict = {}
        if self.can_execute():
            try:
                response_payload = self._request_json(
                    "PUT",
                    f"/{self.index_name}",
                    mapping,
                )
                executed = True
            except RuntimeError as exc:
                message = str(exc)
                if "resource_already_exists_exception" in message:
                    response_payload = {"acknowledged": True, "already_exists": True}
                    executed = True
                else:
                    raise
        return {
            "backend": self.backend_name,
            "executed": executed,
            "index": self.index_name,
            "mapping": mapping,
            "response": response_payload,
        }

    def build_index_mapping(self) -> dict:
        """Return the target Elasticsearch mapping we expect for chunk indexing."""

        return {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {"type": "standard"},
                    }
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "owner_id": {"type": "keyword"},
                    "section_name": {"type": "text"},
                    "content": {"type": "text"},
                    "department_scope": {"type": "keyword"},
                    "visibility_scope": {"type": "keyword"},
                    "role_scope": {"type": "keyword"},
                    "security_level": {"type": "integer"},
                    "current": {"type": "boolean"},
                    "status": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "metadata_json": {"type": "object", "enabled": True},
                }
            },
        }

    def build_bulk_payload(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> str:
        """Build a newline-delimited JSON bulk payload preview for document chunk indexing."""

        lines: list[str] = []
        for chunk in chunks:
            lines.append(json.dumps({"index": {"_index": self.index_name, "_id": chunk.id}}, ensure_ascii=False))
            lines.append(
                json.dumps(
                    {
                        "chunk_id": chunk.id,
                        "doc_id": document.id,
                        "tenant_id": document.tenant_id,
                        "title": document.title,
                        "owner_id": document.owner_id,
                        "section_name": chunk.section_name,
                        "content": chunk.text,
                        "department_scope": chunk.department_scope,
                        "visibility_scope": document.visibility_scope,
                        "role_scope": chunk.role_scope,
                        "security_level": chunk.security_level,
                        "current": document.current,
                        "status": document.status.value,
                        "tags": [tag.lower() for tag in document.tags],
                        "created_at": document.created_at.isoformat(),
                        "updated_at": document.updated_at.isoformat(),
                        "metadata_json": chunk.metadata_json,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(lines)

    def build_search_body(
        self,
        query: str,
        access_filter: AccessFilter,
        terms: Sequence[str],
        top_k: int,
        allowed_chunk_ids: Sequence[str] | None = None,
        tag_filters: Sequence[str] | None = None,
        year_filters: Sequence[int] | None = None,
        exact_terms: Sequence[str] | None = None,
    ) -> dict:
        """Build the Elasticsearch DSL body for a permission-aware keyword search."""

        should_clauses = [
            {"match": {"title": {"query": query, "boost": 3}}},
            {"match": {"section_name": {"query": query, "boost": 2}}},
            {"match": {"content": {"query": query, "boost": 1}}},
        ]
        for term in terms[:8]:
            should_clauses.append({"term": {"title": {"value": term, "boost": 2}}})
            should_clauses.append({"term": {"tags": {"value": term, "boost": 1.2}}})
        for phrase in list(exact_terms or [])[:4]:
            should_clauses.append({"match_phrase": {"title": {"query": phrase, "boost": 4}}})
            should_clauses.append({"match_phrase": {"section_name": {"query": phrase, "boost": 2.5}}})
            should_clauses.append({"match_phrase": {"content": {"query": phrase, "boost": 2}}})

        filters = access_filter.build_elasticsearch_filters(list(allowed_chunk_ids or []))
        if tag_filters:
            filters.append({"terms": {"tags": [tag.lower() for tag in tag_filters]}})
        if year_filters:
            year_should = []
            for year in year_filters:
                year_should.extend(
                    [
                        {"range": {"updated_at": {"gte": f"{year}-01-01", "lt": f"{year + 1}-01-01"}}},
                        {"range": {"created_at": {"gte": f"{year}-01-01", "lt": f"{year + 1}-01-01"}}},
                    ]
                )
            filters.append({"bool": {"should": year_should, "minimum_should_match": 1}})

        return {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
        }

    def _remote_search(
        self,
        query: str,
        terms: Sequence[str],
        candidates: Sequence[tuple[DocumentRecord, DocumentChunk]],
        top_k: int,
        access_filter: AccessFilter | None,
        tag_filters: Sequence[str] | None = None,
        year_filters: Sequence[int] | None = None,
        exact_terms: Sequence[str] | None = None,
    ) -> list[BackendSearchHit]:
        """Execute a real Elasticsearch search and map results back to permission-filtered candidates."""

        candidate_lookup = {chunk.id: (document, chunk) for document, chunk in candidates}
        if access_filter is None:
            raise RuntimeError("Access filter is required for remote Elasticsearch search.")
        body = self.build_search_body(
            query=query,
            access_filter=access_filter,
            terms=terms,
            top_k=top_k,
            allowed_chunk_ids=list(candidate_lookup.keys()),
            tag_filters=tag_filters,
            year_filters=year_filters,
            exact_terms=exact_terms,
        )
        payload = self._request_json("POST", f"/{self.index_name}/_search", body)
        hits: list[BackendSearchHit] = []
        for hit in payload.get("hits", {}).get("hits", []):
            chunk_id = hit.get("_id") or hit.get("_source", {}).get("chunk_id")
            if chunk_id not in candidate_lookup:
                continue
            document, chunk = candidate_lookup[chunk_id]
            hits.append(
                BackendSearchHit(
                    document=document,
                    chunk=chunk,
                    score=float(hit.get("_score", 0.0)),
                    backend=self.backend_name,
                    matched_terms=list(terms),
                )
            )
        return hits

    def _bulk_upsert(self, document: DocumentRecord, chunks: Sequence[DocumentChunk]) -> dict:
        """Execute a real Elasticsearch bulk indexing request."""

        payload = self.build_bulk_payload(document, chunks)
        return self._request_json(
            "POST",
            "/_bulk?refresh=wait_for",
            raw_body=(payload + "\n").encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )

    def _bulk_delete(self, chunks: Sequence[DocumentChunk]) -> dict:
        """Execute a real Elasticsearch bulk delete request."""

        payload = "\n".join(
            json.dumps({"delete": {"_index": self.index_name, "_id": chunk.id}}, ensure_ascii=False)
            for chunk in chunks
        )
        return self._request_json(
            "POST",
            "/_bulk?refresh=wait_for",
            raw_body=(payload + "\n").encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        raw_body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Send a JSON HTTP request to Elasticsearch and decode the response body."""

        if not self.endpoint:
            raise RuntimeError("Elasticsearch endpoint is not configured.")

        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        body = raw_body
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        url = self.endpoint.rstrip("/") + path
        req = request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with request.urlopen(req, timeout=5) as response:
                response_body = response.read().decode("utf-8").strip()
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(message or str(exc)) from exc

        if not response_body:
            return {}
        return json.loads(response_body)
