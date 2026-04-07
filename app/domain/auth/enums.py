from enum import Enum


class RoleScope(str, Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"
    SECURITY_ADMIN = "security_admin"
