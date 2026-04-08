from __future__ import annotations

from pathlib import Path


class LocalDocumentSourceStore:
    """Persists uploaded source files locally so ingestion can continue outside the request path."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_source(self, doc_id: str, source_type: str, content: bytes) -> str:
        """Write the original upload bytes to a deterministic staging path and return that path."""

        target = self._resolve_path(doc_id, source_type)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    def read_source(self, doc_id: str, source_type: str) -> bytes:
        """Read staged upload bytes back into memory for parsing and index building."""

        return self._resolve_path(doc_id, source_type).read_bytes()

    def delete_source(self, doc_id: str, source_type: str) -> None:
        """Delete one staged upload after cleanup or explicit retention management."""

        self._resolve_path(doc_id, source_type).unlink(missing_ok=True)

    def has_source(self, doc_id: str, source_type: str) -> bool:
        """Return whether the staged source exists for one document version."""

        return self._resolve_path(doc_id, source_type).exists()

    def _resolve_path(self, doc_id: str, source_type: str) -> Path:
        suffix = self._suffix_for_source_type(source_type)
        return self.base_dir / f"{doc_id}{suffix}"

    @staticmethod
    def _suffix_for_source_type(source_type: str) -> str:
        normalized = (source_type or "manual").lower()
        if normalized == "markdown":
            return ".md"
        if normalized in {"html", "pdf", "docx"}:
            return f".{normalized}"
        return ".txt"
