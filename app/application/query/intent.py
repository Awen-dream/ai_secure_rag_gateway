def classify_query_intent(query: str) -> str:
    normalized = query.lower()
    if any(token in normalized for token in ["编号", "id", "version", "v", "第几版"]):
        return "exact_lookup"
    if any(token in normalized for token in ["总结", "归纳", "对比"]):
        return "summary"
    return "standard_qa"
