from app.domain.auth.models import UserContext
from app.domain.risk.models import PolicyDefinition, RiskAction
from app.domain.risk.rules import baseline_policy
from app.infrastructure.db.repositories.memory import store


class PolicyEngine:
    def __init__(self) -> None:
        if not store.policies:
            policy = baseline_policy()
            store.policies[policy.id] = policy

    def list_policies(self) -> list[PolicyDefinition]:
        return list(store.policies.values())

    def add_policy(self, policy: PolicyDefinition) -> PolicyDefinition:
        store.policies[policy.id] = policy
        return policy

    def evaluate(self, user: UserContext, query: str, matched_chunks: int) -> tuple[RiskAction, str]:
        normalized = query.lower()
        for policy in store.policies.values():
            if not policy.enabled:
                continue
            if any(term in normalized for term in policy.high_risk_terms):
                return RiskAction.REFUSE, "high"
            if user.department_id in policy.restricted_departments and matched_chunks == 0:
                return RiskAction.CITATIONS_ONLY, "medium"
        return RiskAction.ALLOW, "low"
