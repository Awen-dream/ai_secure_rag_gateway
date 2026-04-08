from __future__ import annotations

import re

from app.domain.auth.models import UserContext
from app.domain.citations.services import Citation
from app.domain.risk.models import OutputGuardResult, RiskAction


EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
CN_ID_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
BANK_CARD_PATTERN = re.compile(r"(?<!\d)\d{16,19}(?!\d)")


class OutputGuard:
    """Applies post-generation masking, downgrade and refusal rules before responses leave the gateway."""

    secret_terms = (
        "api key",
        "secret key",
        "access token",
        "refresh token",
        "password",
        "private key",
        "数据库密码",
        "密钥",
        "令牌",
        "口令",
    )
    restricted_detail_terms = (
        "身份证",
        "银行卡",
        "手机号",
        "邮箱",
        "薪资",
        "工资",
        "合同金额",
    )

    def apply(
        self,
        user: UserContext,
        answer: str,
        citations: list[Citation],
        risk_action: RiskAction,
    ) -> OutputGuardResult:
        """Apply output-side safety controls and return the final action plus sanitized answer."""

        if risk_action == RiskAction.REFUSE:
            return OutputGuardResult(action=RiskAction.REFUSE, answer=answer, risk_level="high", reasons=["input_policy"])

        normalized = answer.lower()
        if user.role not in {"admin", "security_admin"} and any(term in normalized for term in self.secret_terms):
            return OutputGuardResult(
                action=RiskAction.REFUSE,
                answer="结论：当前回答包含高敏凭据或密钥信息，平台已拒绝输出。\n依据：输出安全策略禁止返回密钥、口令或令牌类内容。",
                risk_level="high",
                reasons=["secret_material"],
            )

        masked_answer, masking_reasons = self._mask_sensitive_content(answer)
        if masking_reasons:
            return OutputGuardResult(
                action=RiskAction.MASK,
                answer=masked_answer,
                risk_level="medium",
                reasons=masking_reasons,
            )

        if risk_action == RiskAction.CITATIONS_ONLY or (
            user.department_id in {"hr", "finance", "legal"} and any(term in answer for term in self.restricted_detail_terms)
        ):
            citation_lines = [
                f"[{item.index}] {item.title} / {item.section_name} / v{item.version}"
                for item in citations
            ]
            citation_block = "\n".join(citation_lines) if citation_lines else "无"
            return OutputGuardResult(
                action=RiskAction.CITATIONS_ONLY,
                answer=(
                    "结论：当前问题涉及高敏领域，平台已降级为仅返回可核验引用。\n"
                    f"引用来源：\n{citation_block}"
                ),
                risk_level="medium",
                reasons=["citations_only"],
            )

        return OutputGuardResult(action=RiskAction.ALLOW, answer=answer, risk_level="low")

    def _mask_sensitive_content(self, answer: str) -> tuple[str, list[str]]:
        """Mask common PII patterns such as email, mobile phone, ID card and bank card."""

        masked = answer
        reasons: list[str] = []

        masked, changed = EMAIL_PATTERN.subn("[REDACTED_EMAIL]", masked)
        if changed:
            reasons.append("email")

        masked, changed = PHONE_PATTERN.subn("[REDACTED_PHONE]", masked)
        if changed:
            reasons.append("phone")

        masked, changed = CN_ID_PATTERN.subn("[REDACTED_CN_ID]", masked)
        if changed:
            reasons.append("cn_id")

        masked, changed = BANK_CARD_PATTERN.subn("[REDACTED_BANK_CARD]", masked)
        if changed:
            reasons.append("bank_card")

        return masked, reasons
