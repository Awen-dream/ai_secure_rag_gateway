from fastapi import APIRouter, Depends

from app.core.security import get_current_user
from app.domain.auth.models import UserContext

router = APIRouter()


@router.get("/me")
def me(user: UserContext = Depends(get_current_user)) -> dict:
    return user.dict()
