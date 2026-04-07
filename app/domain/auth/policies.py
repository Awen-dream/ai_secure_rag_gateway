from app.domain.auth.models import UserContext


def can_access_department(target_departments: list[str], user: UserContext) -> bool:
    return not target_departments or user.department_id in target_departments
