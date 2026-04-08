from app.domain.auth.models import UserContext


def can_access_department(target_departments: list[str], user: UserContext) -> bool:
    return not target_departments or user.department_id in target_departments


def can_access_visibility(visibility_scope: list[str], owner_id: str, department_scope: list[str], user: UserContext) -> bool:
    if not visibility_scope:
        return True
    if "public" in visibility_scope or "tenant" in visibility_scope:
        return True
    if "owner" in visibility_scope and owner_id == user.user_id:
        return True
    if "department" in visibility_scope:
        return can_access_department(department_scope, user)
    return False
