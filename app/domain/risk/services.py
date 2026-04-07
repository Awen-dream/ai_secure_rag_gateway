from app.domain.auth.models import UserContext
from app.domain.risk.models import PolicyDefinition, RiskAction
from app.domain.risk.rules import baseline_policy
from app.infrastructure.db.repositories.base import MetadataRepository


class PolicyEngine:
    def __init__(self, repository: MetadataRepository) -> None:
        self.repository = repository
        if not self.repository.list_policies():
            self.repository.save_policy(baseline_policy())

    def list_policies(self) -> list[PolicyDefinition]:
        return self.repository.list_policies()

    def add_policy(self, policy: PolicyDefinition) -> PolicyDefinition:
        self.repository.save_policy(policy)
        return policy

    def evaluate(self, user: UserContext, query: str, matched_chunks: int) -> tuple[RiskAction, str]:
        normalized = query.lower()
        for policy in self.repository.list_policies():
            if not policy.enabled:
                continue
            if any(term in normalized for term in policy.high_risk_terms):
                return RiskAction.REFUSE, "high"
            if user.department_id in policy.restricted_departments and matched_chunks == 0:
                return RiskAction.CITATIONS_ONLY, "medium"
        return RiskAction.ALLOW, "low"
