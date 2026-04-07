from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_document_service
from app.core.security import get_current_user
from app.domain.auth.models import UserContext
from app.domain.documents.models import DocumentRecord
from app.domain.documents.schemas import DocumentUploadRequest
from app.domain.documents.services import DocumentService

router = APIRouter()


@router.post("/upload", response_model=DocumentRecord)
def upload_document(
    payload: DocumentUploadRequest,
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentRecord:
    return service.upload_document(payload, user)


@router.get("", response_model=list[DocumentRecord])
def list_documents(
    user: UserContext = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentRecord]:
    return service.list_documents(user)


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
