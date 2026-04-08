from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")
_COMPARISON_PATTERNS = (
    re.compile(r"(?:对比|比较|区别|差异|不同|变化|优缺点)"),
    re.compile(r"\b(?:compare|comparison|difference|diff|versus|vs)\b"),
)
_SUMMARY_PATTERNS = (
    re.compile(r"(?:总结|总结一下|归纳|概述|概括|摘要|梳理|总览|汇总)"),
    re.compile(r"\b(?:summary|summarize|overview|recap|outline)\b"),
)
_LIST_PATTERNS = (
    re.compile(r"(?:列出|列举|有哪些|包括哪些|分别是什?么)"),
)
_EXACT_VERSION_PATTERNS = (
    re.compile(r"第\s*\d+\s*版"),
    re.compile(r"\bversion\s*(?:no\.?\s*)?\d+(?:\.\d+)*\b"),
    re.compile(r"\brev(?:ision)?\s*(?:no\.?\s*)?\d+(?:\.\d+)*\b"),
    re.compile(r"\bv\s*\d+(?:\.\d+)*\b"),
)
_EXACT_METADATA_PATTERNS = (
    re.compile(r"(?:文档编号|文件编号|单号|流水号|source_document_id|document_id|文档id|版本号|修订版)"),
    re.compile(r"\b(?:doc(?:ument)?\s*id|request\s*id|ticket\s*id|version)\b"),
)
_LOOKUP_ACTION_PATTERNS = (
    re.compile(r"(?:查询|查找|查下|查一下|查一查|定位|找到|查看|看下|看一下|获取|返回|给我)"),
    re.compile(r"(?:哪一版|第几版|哪个版本|对应.*编号|对应.*id|是什么|是多少|在哪|在哪里|当前版本|最新版本)"),
    re.compile(r"\b(?:find|lookup|show|get|which|where|what)\b"),
)


@dataclass(frozen=True)
class IntentClassification:
    intent: str
    confidence: float
    reasons: list[str]


def _normalize_query(query: str) -> str:
    return _WHITESPACE_RE.sub(" ", query.strip().lower())


def _matches_any(patterns: tuple[re.Pattern[str], ...], normalized: str) -> bool:
    return any(pattern.search(normalized) for pattern in patterns)


def classify_query_intent_details(query: str) -> IntentClassification:
    """Classify retrieval intent with low-cost, explainable heuristics.

    The classifier intentionally stays rule-based so it remains deterministic,
    cheap to run on every request, and easy to tune from query logs. We bias
    toward avoiding false positives on `exact_lookup`, because the retrieval
    profile for exact matching is stricter than the default QA profile.
    """

    normalized = _normalize_query(query)
    if not normalized:
        return IntentClassification(intent="standard_qa", confidence=0.35, reasons=["empty_or_blank_query"])

    comparison_match = _matches_any(_COMPARISON_PATTERNS, normalized)
    summary_match = _matches_any(_SUMMARY_PATTERNS, normalized)
    list_match = _matches_any(_LIST_PATTERNS, normalized)
    version_match = _matches_any(_EXACT_VERSION_PATTERNS, normalized)
    metadata_match = _matches_any(_EXACT_METADATA_PATTERNS, normalized)
    lookup_match = _matches_any(_LOOKUP_ACTION_PATTERNS, normalized)

    summary_score = 0
    exact_score = 0
    reasons: list[str] = []

    if comparison_match:
        summary_score += 3
        reasons.append("comparison_language")
    if summary_match:
        summary_score += 2
        reasons.append("summary_language")
    if list_match:
        summary_score += 1
        reasons.append("enumeration_language")

    if version_match:
        exact_score += 3
        reasons.append("explicit_version_pattern")
    if metadata_match and lookup_match:
        exact_score += 3
        reasons.append("metadata_lookup_pattern")
    elif metadata_match:
        exact_score += 1
        reasons.append("metadata_reference")

    # Comparison-style requests usually benefit from broader recall even when
    # version numbers are present, such as "v2 和 v3 有什么区别".
    if summary_score >= 3:
        confidence = 0.92 if summary_score >= 4 else 0.84
        return IntentClassification(intent="summary", confidence=confidence, reasons=reasons)
    if exact_score >= 3:
        confidence = 0.9 if exact_score >= 5 else 0.82
        return IntentClassification(intent="exact_lookup", confidence=confidence, reasons=reasons)
    if summary_score >= 2:
        return IntentClassification(intent="summary", confidence=0.72, reasons=reasons)
    fallback_reasons = reasons or ["default_standard_qa"]
    return IntentClassification(intent="standard_qa", confidence=0.58, reasons=fallback_reasons)


def classify_query_intent(query: str) -> str:
    return classify_query_intent_details(query).intent
