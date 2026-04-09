from app.application.query.planning import QueryPlanningService


def prepare_query(query: str) -> dict:
    """Legacy helper kept for lightweight tooling that needs query-planning output as plain dict."""

    result = QueryPlanningService().plan(query)
    understanding = result.understanding
    return {
        "rewritten_query": result.rewritten_query,
        "intent": understanding.intent,
        "intent_confidence": understanding.confidence,
        "intent_reasons": understanding.reasons,
        "understanding_source": understanding.source,
        "rule_rewritten_query": understanding.rule_rewritten_query,
        "rule_intent": understanding.rule_intent,
        "rule_intent_confidence": understanding.rule_confidence,
        "rule_intent_reasons": understanding.rule_reasons,
    }
