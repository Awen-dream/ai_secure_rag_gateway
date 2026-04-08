from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.application.ingestion.document_parser import infer_source_type, normalize_text_content
from app.api.deps import get_document_ingestion_orchestrator, get_document_service, get_document_task_queue
from app.core.security import get_current_user
from app.application.ingestion.orchestrator import DocumentIngestionOrchestrator
from app.domain.auth.models import UserContext
from app.domain.documents.models import DocumentRecord
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.documents.services import DocumentService
from app.infrastructure.queue.worker import DocumentIngestionTaskQueue

router = APIRouter()


def _parse_csv_field(value: Optional[str], default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


@router.post("/upload", response_model=DocumentRecord)
def upload_document(
    payload: DocumentUploadRequest,
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    orchestrator: DocumentIngestionOrchestrator = Depends(get_document_ingestion_orchestrator),
    task_queue: DocumentIngestionTaskQueue = Depends(get_document_task_queue),
) -> DocumentRecord:
    normalized_payload = payload.model_copy(
        update={
            "source_type": infer_source_type(declared_type=payload.source_type),
            "content": payload.content if payload.async_mode else normalize_text_content(payload.content, payload.source_type),
        }
    )
    document = service.upload_document(normalized_payload, user)
    if normalized_payload.async_mode:
        task_queue.enqueue_document(document.id)
    else:
        document = orchestrator.process_document(document.id)
    return document


@router.post("/upload-file", response_model=DocumentRecord)
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    source_type: Optional[str] = Form(default=None),
    source_uri: Optional[str] = Form(default=None),
    source_connector: Optional[str] = Form(default=None),
    source_document_id: Optional[str] = Form(default=None),
    source_document_version: Optional[str] = Form(default=None),
    owner_id: Optional[str] = Form(default=None),
    department_scope: Optional[str] = Form(default=None),
    visibility_scope: Optional[str] = Form(default=None),
    security_level: int = Form(default=1),
    tags: Optional[str] = Form(default=None),
    async_mode: bool = Form(default=False),
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    orchestrator: DocumentIngestionOrchestrator = Depends(get_document_ingestion_orchestrator),
    task_queue: DocumentIngestionTaskQueue = Depends(get_document_task_queue),
) -> DocumentRecord:
    try:
        file_bytes = await file.read()
        normalized_source_type = infer_source_type(file.filename, source_type)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    payload = DocumentUploadRequest(
        title=title or file.filename or "uploaded-document",
        content="",
        source_type=normalized_source_type,
        source_uri=source_uri,
        source_connector=source_connector,
        source_document_id=source_document_id,
        source_document_version=source_document_version,
        owner_id=owner_id,
        department_scope=_parse_csv_field(department_scope),
        visibility_scope=_parse_csv_field(visibility_scope, default=["tenant"]),
        security_level=security_level,
        tags=_parse_csv_field(tags),
        async_mode=async_mode,
    )
    document = service.upload_document_file(payload, user, file_bytes=file_bytes, process_async=async_mode)
    if async_mode:
        task_queue.enqueue_document(document.id)
    else:
        document = orchestrator.process_document(document.id)
    return document


@router.get("", response_model=list[DocumentRecord])
def list_documents(
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentRecord]:
    return service.list_documents(user)


@router.get("/{doc_id}", response_model=DocumentRecord)
def get_document(
    doc_id: str,
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentRecord:
    try:
        return service.get_document(doc_id, user)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.") from exc


@router.post("/{doc_id}/reindex", response_model=DocumentRecord)
def reindex_document(
    doc_id: str,
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentRecord:
    try:
        return service.reindex_document(doc_id, user)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.") from exc


@router.post("/{doc_id}/retry", response_model=DocumentRecord)
def retry_document(
    doc_id: str,
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    task_queue: DocumentIngestionTaskQueue = Depends(get_document_task_queue),
) -> DocumentRecord:
    try:
        document = service.retry_document(doc_id, user)
        task_queue.enqueue_document(document.id)
        return document
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.") from exc
