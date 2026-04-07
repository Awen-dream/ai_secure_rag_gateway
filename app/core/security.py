from fastapi import Depends, Header, HTTPException, status

from app.domain.auth.models import UserContext


def get_current_user(
    x_user_id: str = Header(..., alias="X-User-Id"),
    x_tenant_id: str = Header(..., alias="X-Tenant-Id"),
    x_department_id: str = Header(..., alias="X-Department-Id"),
    x_role: str = Header(..., alias="X-Role"),
    x_clearance_level: int = Header(..., alias="X-Clearance-Level"),
) -> UserContext:
    return UserContext(
        user_id=x_user_id,
        tenant_id=x_tenant_id,
        department_id=x_department_id,
        role=x_role,
        clearance_level=x_clearance_level,
    )


def require_admin(user: UserContext = Depends(get_current_user)) -> UserContext:
    if user.role.lower() not in {"admin", "security_admin", "platform_admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges are required for this endpoint.",
        )
    return user
