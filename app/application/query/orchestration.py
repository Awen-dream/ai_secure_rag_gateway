from app.application.query.intent import classify_query_intent_details
from app.application.query.rewrite import rewrite_query


def prepare_query(query: str) -> dict:
    rewritten = rewrite_query(query)
    intent_result = classify_query_intent_details(rewritten)
    return {
        "rewritten_query": rewritten,
        "intent": intent_result.intent,
        "intent_confidence": intent_result.confidence,
        "intent_reasons": intent_result.reasons,
    }
