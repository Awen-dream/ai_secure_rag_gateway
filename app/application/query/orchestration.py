from app.application.query.intent import classify_query_intent
from app.application.query.rewrite import rewrite_query


def prepare_query(query: str) -> dict:
    rewritten = rewrite_query(query)
    return {"rewritten_query": rewritten, "intent": classify_query_intent(rewritten)}
