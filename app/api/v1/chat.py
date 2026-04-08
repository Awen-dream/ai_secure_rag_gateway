import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service, get_rate_limit_service
from app.core.security import get_current_user
from app.domain.auth.models import UserContext
from app.domain.chat.schemas import ChatQueryRequest, ChatQueryResponse
from app.domain.chat.services import ChatService
from app.domain.risk.rate_limit import RateLimitService

router = APIRouter()


@router.post("/query", response_model=ChatQueryResponse)
def query_chat(
    payload: ChatQueryRequest,
    user: UserContext = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
    rate_limiter: RateLimitService = Depends(get_rate_limit_service),
) -> ChatQueryResponse:
    allowed, _ = rate_limiter.check_user(user.user_id, scope="chat.query")
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    try:
        return service.query(payload, user)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.") from exc


@router.post("/stream")
def stream_chat(
    payload: ChatQueryRequest,
    user: UserContext = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
    rate_limiter: RateLimitService = Depends(get_rate_limit_service),
) -> StreamingResponse:
    allowed, _ = rate_limiter.check_user(user.user_id, scope="chat.stream")
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")
    response = service.query(payload, user)

    def event_stream():
        yield f"data: {json.dumps({'event': 'start', 'request_id': response.request_id}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event': 'answer', 'answer': response.answer}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event': 'citations', 'citations': [citation.dict() for citation in response.citations]}, ensure_ascii=False)}\n\n"
        yield "data: {\"event\": \"end\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/sessions")
def list_sessions(
    user: UserContext = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
) -> list[dict]:
    return [session.dict() for session in service.list_sessions(user)]


@router.get("/sessions/{session_id}")
def get_session_detail(
    session_id: str,
    user: UserContext = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
) -> dict:
    try:
        detail = service.get_session_detail(session_id, user)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.") from exc
    return {
        "session": detail["session"].dict(),
        "messages": [message.dict() for message in detail["messages"]],
    }
