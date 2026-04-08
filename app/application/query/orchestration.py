from app.application.query.understanding import QueryUnderstandingService


def prepare_query(query: str) -> dict:
    result = QueryUnderstandingService().understand(query)
    return {
        "rewritten_query": result.rewritten_query,
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "intent_reasons": result.reasons,
        "understanding_source": result.source,
        "rule_rewritten_query": result.rule_rewritten_query,
        "rule_intent": result.rule_intent,
        "rule_intent_confidence": result.rule_confidence,
        "rule_intent_reasons": result.rule_reasons,
    }
