from app.domain.risk.models import PolicyDefinition


def baseline_policy() -> PolicyDefinition:
    return PolicyDefinition(
        id="policy_secure_gateway_v1",
        name="Secure Enterprise Gateway Baseline",
        description="Detect prompt injection and apply conservative behavior for sensitive departments.",
        high_risk_terms=[
            "ignore previous instructions",
            "system prompt",
            "reveal hidden",
            "export all data",
            "dump database",
        ],
        restricted_departments=["hr", "finance", "legal"],
    )
