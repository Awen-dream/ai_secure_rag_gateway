import unittest

from app.domain.auth.models import UserContext
from app.domain.citations.services import Citation
from app.domain.risk.models import RiskAction
from app.domain.risk.output_guard import OutputGuard


def build_user(role: str = "employee", department_id: str = "engineering") -> UserContext:
    return UserContext(
        user_id="u1",
        tenant_id="t1",
        department_id=department_id,
        role=role,
        clearance_level=2,
    )


def build_citations() -> list[Citation]:
    return [
        Citation(
            index=1,
            doc_id="doc_1",
            title="报销制度",
            section_name="Section 1",
            version=1,
            source_uri=None,
        )
    ]


class OutputGuardTest(unittest.TestCase):
    def test_masks_common_pii_patterns(self) -> None:
        guard = OutputGuard()
        result = guard.apply(
            user=build_user(),
            answer="联系人邮箱 test@example.com，手机号 13812345678，身份证 11010519491231002X。",
            citations=build_citations(),
            risk_action=RiskAction.ALLOW,
        )

        self.assertEqual(result.action, RiskAction.MASK)
        self.assertIn("[REDACTED_EMAIL]", result.answer)
        self.assertIn("[REDACTED_PHONE]", result.answer)
        self.assertIn("[REDACTED_CN_ID]", result.answer)
        self.assertEqual(result.risk_level, "medium")

    def test_refuses_secret_material_for_non_admin(self) -> None:
        guard = OutputGuard()
        result = guard.apply(
            user=build_user(),
            answer="系统数据库密码是 secret-password-123。",
            citations=build_citations(),
            risk_action=RiskAction.ALLOW,
        )

        self.assertEqual(result.action, RiskAction.REFUSE)
        self.assertIn("平台已拒绝输出", result.answer)
        self.assertEqual(result.risk_level, "high")

    def test_high_sensitive_departments_downgrade_to_citations_only(self) -> None:
        guard = OutputGuard()
        result = guard.apply(
            user=build_user(department_id="finance"),
            answer="员工薪资明细为 50000 元。",
            citations=build_citations(),
            risk_action=RiskAction.ALLOW,
        )

        self.assertEqual(result.action, RiskAction.CITATIONS_ONLY)
        self.assertIn("仅返回可核验引用", result.answer)
        self.assertIn("[1] 报销制度 / Section 1 / v1", result.answer)


if __name__ == "__main__":
    unittest.main()
