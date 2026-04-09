from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.auth.models import UserContext
from app.domain.auth.policies import can_access_department, can_access_visibility
from app.domain.documents.models import DocumentChunk, DocumentRecord


class AccessFilter(BaseModel):
    """Request-scoped access filter shared by reads and retrieval backends."""

    tenant_id: str
    user_id: str
    department_id: str
    role: str
    max_security_level: int
    visibility_scope: list[str] = Field(default_factory=lambda: ["public", "tenant", "department", "owner"])

    def allows_document(self, document: DocumentRecord) -> bool:
        """Return whether one document is visible under the current access scope."""

        return (
            document.current
            and document.tenant_id == self.tenant_id
            and document.security_level <= self.max_security_level
            and can_access_department(document.department_scope, self.to_user_context())
            and can_access_visibility(document.visibility_scope, document.owner_id, document.department_scope, self.to_user_context())
        )

    def allows_chunk(self, chunk: DocumentChunk) -> bool:
        """Return whether one chunk may participate in retrieval under the current access scope."""

        role_ok = not chunk.role_scope or self.role in chunk.role_scope
        department_ok = can_access_department(chunk.department_scope, self.to_user_context())
        return role_ok and department_ok and chunk.security_level <= self.max_security_level

    def build_elasticsearch_filters(self, allowed_chunk_ids: list[str] | None = None) -> list[dict]:
        """Build Elasticsearch filters aligned with tenant and access boundaries."""

        filters: list[dict] = [
            {"term": {"tenant_id": self.tenant_id}},
            {"term": {"current": True}},
            {"range": {"security_level": {"lte": self.max_security_level}}},
            {
                "bool": {
                    "should": [
                        {"term": {"visibility_scope": "public"}},
                        {"term": {"visibility_scope": "tenant"}},
                        {
                            "bool": {
                                "must": [
                                    {"term": {"visibility_scope": "owner"}},
                                    {"term": {"owner_id": self.user_id}},
                                ]
                            }
                        },
                        {
                            "bool": {
                                "must": [
                                    {"term": {"visibility_scope": "department"}},
                                    {
                                        "bool": {
                                            "should": [
                                                {"term": {"department_scope": self.department_id}},
                                                {"bool": {"must_not": [{"exists": {"field": "department_scope"}}]}},
                                            ],
                                            "minimum_should_match": 1,
                                        }
                                    },
                                ]
                            }
                        },
                        {"bool": {"must_not": [{"exists": {"field": "visibility_scope"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            {
                "bool": {
                    "should": [
                        {"term": {"department_scope": self.department_id}},
                        {"bool": {"must_not": [{"exists": {"field": "department_scope"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            {
                "bool": {
                    "should": [
                        {"term": {"role_scope": self.role}},
                        {"bool": {"must_not": [{"exists": {"field": "role_scope"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        ]
        if allowed_chunk_ids:
            filters.append({"terms": {"chunk_id": allowed_chunk_ids}})
        return filters

    def build_pgvector_params(self, allowed_chunk_ids: list[str]) -> dict:
        """Build the parameter bundle used by pgvector search SQL templates."""

        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "department_id": self.department_id,
            "role": self.role,
            "max_security_level": self.max_security_level,
            "chunk_ids": allowed_chunk_ids,
        }

    def build_pgvector_where_clause(self) -> str:
        """Build the permission-aware SQL predicate shared by pgvector planning and execution."""

        return (
            "tenant_id = %(tenant_id)s "
            "AND current = TRUE "
            "AND security_level <= %(max_security_level)s "
            "AND chunk_id = ANY(%(chunk_ids)s::text[]) "
            "AND (jsonb_array_length(role_scope) = 0 OR role_scope ? %(role)s) "
            "AND (jsonb_array_length(department_scope) = 0 OR department_scope ? %(department_id)s) "
            "AND ("
            "visibility_scope ? 'public' "
            "OR visibility_scope ? 'tenant' "
            "OR (visibility_scope ? 'owner' AND owner_id = %(user_id)s) "
            "OR (visibility_scope ? 'department' AND (jsonb_array_length(department_scope) = 0 OR department_scope ? %(department_id)s))"
            ")"
        )

    def to_user_context(self) -> UserContext:
        """Convert the access filter back into user-context shape for shared policy helpers."""

        return UserContext(
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            department_id=self.department_id,
            role=self.role,
            clearance_level=self.max_security_level,
        )


def build_access_filter(user: UserContext) -> AccessFilter:
    """Construct one access filter from the authenticated user context."""

    return AccessFilter(
        tenant_id=user.tenant_id,
        user_id=user.user_id,
        department_id=user.department_id,
        role=user.role,
        max_security_level=user.clearance_level,
    )


def build_access_signature(user: UserContext) -> str:
    """Render one compact access signature so session state resets on auth changes."""

    return "|".join(
        [
            user.tenant_id,
            user.user_id,
            user.department_id,
            user.role,
            str(user.clearance_level),
        ]
    )
