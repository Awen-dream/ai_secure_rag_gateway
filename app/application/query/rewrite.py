from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

_FOLLOW_UP_MARKER_RE = re.compile(r"(?:这个|那个|它|上面的|继续|刚才|上一轮|前面|这里|其中|呢[？?]?)")
_POLITE_PREFIX_RE = re.compile(r"^(?:请问一下|请问|帮我看下|帮我看看|帮我查下|帮我查查|麻烦你|麻烦|想问下|想了解下)\s*")
_POLITE_SUFFIX_RE = re.compile(r"(?:谢谢|麻烦了)[！!。.\s]*$")
_TAG_FILTER_RE = re.compile(r"(?:#|标签[:：]\s*|tag[:：]\s*)([a-zA-Z0-9_\-\u4e00-\u9fff]{1,32})")
_YEAR_FILTER_RE = re.compile(r"(?<!\d)(20\d{2})(?:年)?")
_RECENCY_HINT_RE = re.compile(r"(?:最新|最近|当前|现行|今年|本月|本周|新版本|最新版)")
_QUOTED_PHRASE_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]{2,}")
_CONTROL_FRAGMENT_RE = re.compile(r"(?:标签[:：]\s*[a-zA-Z0-9_\-\u4e00-\u9fff]{1,32}|#[a-zA-Z0-9_\-\u4e00-\u9fff]{1,32})")
_STOPWORDS = {
    "什么",
    "多少",
    "一下",
    "一下子",
    "请问",
    "帮我",
    "看看",
    "查下",
    "查查",
    "一个",
    "哪个",
    "哪些",
    "是否",
    "怎么",
    "如何",
    "为啥",
    "为什么",
    "以及",
    "还有",
    "关于",
}
_SYNONYM_MAP = {
    "报销": ["费用报销", "报销申请"],
    "采购": ["采购申请", "采购流程"],
    "制度": ["规范", "办法"],
    "手册": ["指南", "说明"],
    "审批": ["审批流程", "审批要求"],
    "时限": ["时效", "办理时限", "审批时限"],
    "权限": ["访问权限", "授权范围"],
    "版本": ["版次", "修订版"],
    "生效": ["生效日期", "执行时间"],
}
_PHRASE_EXPANSIONS = [
    (re.compile(r"(?:审批|处理|办理).*(?:时限|时效)|(?:时限|时效).*(?:审批|处理|办理)"), ["审批时限", "审批时效", "办理时限"]),
    (re.compile(r"(?:文档|文件).*(?:编号|id)|(?:编号|id).*(?:文档|文件)", re.IGNORECASE), ["文档编号", "文件编号", "document id"]),
    (re.compile(r"(?:最新|现行).*(?:制度|版本)|(?:制度|版本).*(?:最新|现行)"), ["最新版本", "现行版本", "当前版本"]),
]


@dataclass(frozen=True)
class QueryRewritePlan:
    original_query: str
    rewritten_query: str
    keywords: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    exact_phrases: list[str] = field(default_factory=list)
    tag_filters: list[str] = field(default_factory=list)
    year_filters: list[int] = field(default_factory=list)
    recency_hint: bool = False


def rewrite_query(
    query: str,
    last_user_query: Optional[str] = None,
    session_summary: Optional[str] = None,
) -> str:
    return build_query_rewrite_plan(query, last_user_query=last_user_query, session_summary=session_summary).rewritten_query


def build_query_rewrite_plan(
    query: str,
    last_user_query: Optional[str] = None,
    session_summary: Optional[str] = None,
) -> QueryRewritePlan:
    normalized_original = _normalize_query_text(query)
    contextual = normalized_original
    if contextual and _FOLLOW_UP_MARKER_RE.search(contextual):
        if last_user_query:
            contextual = _normalize_query_text(f"{last_user_query} {contextual}")
        elif session_summary:
            contextual = _normalize_query_text(f"{session_summary[:120]} {contextual}")

    cleaned = _cleanup_query_text(contextual)
    exact_phrases = _extract_exact_phrases(cleaned)
    tag_filters = _extract_tag_filters(normalized_original)
    year_filters = _extract_year_filters(normalized_original)
    recency_hint = bool(_RECENCY_HINT_RE.search(normalized_original))
    keywords = _extract_keywords(cleaned)
    expanded_terms = _expand_terms(cleaned, keywords, exact_phrases)
    rewritten = cleaned or normalized_original
    return QueryRewritePlan(
        original_query=normalized_original,
        rewritten_query=rewritten,
        keywords=keywords,
        expanded_terms=expanded_terms,
        exact_phrases=exact_phrases,
        tag_filters=tag_filters,
        year_filters=year_filters,
        recency_hint=recency_hint,
    )


def refine_query_rewrite_plan(plan: QueryRewritePlan, rewritten_query: str) -> QueryRewritePlan:
    normalized_rewritten = _cleanup_query_text(rewritten_query) or plan.rewritten_query
    exact_phrases = list(dict.fromkeys([*plan.exact_phrases, *_extract_exact_phrases(normalized_rewritten)]))
    keywords = list(dict.fromkeys([*_extract_keywords(normalized_rewritten), *plan.keywords]))
    expanded_terms = list(
        dict.fromkeys([*_expand_terms(normalized_rewritten, keywords, exact_phrases), *plan.expanded_terms])
    )
    return QueryRewritePlan(
        original_query=plan.original_query,
        rewritten_query=normalized_rewritten,
        keywords=keywords,
        expanded_terms=expanded_terms,
        exact_phrases=exact_phrases,
        tag_filters=list(plan.tag_filters),
        year_filters=list(plan.year_filters),
        recency_hint=plan.recency_hint or bool(_RECENCY_HINT_RE.search(normalized_rewritten)),
    )


def _normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.replace("?", "？").replace("!", "！")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _cleanup_query_text(text: str) -> str:
    cleaned = _POLITE_PREFIX_RE.sub("", _normalize_query_text(text))
    cleaned = _POLITE_SUFFIX_RE.sub("", cleaned)
    cleaned = _CONTROL_FRAGMENT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_exact_phrases(query: str) -> list[str]:
    phrases = [match.strip() for match in _QUOTED_PHRASE_RE.findall(query) if match.strip()]
    if not phrases and len(query) <= 24 and " " not in query and len(query) >= 4:
        phrases.append(query)
    return list(dict.fromkeys(phrases))


def _extract_tag_filters(query: str) -> list[str]:
    return list(dict.fromkeys(match.lower() for match in _TAG_FILTER_RE.findall(query)))


def _extract_year_filters(query: str) -> list[int]:
    years: list[int] = []
    for match in _YEAR_FILTER_RE.findall(query):
        try:
            year = int(match)
        except ValueError:
            continue
        if 2000 <= year <= 2100:
            years.append(year)
    return list(dict.fromkeys(years))


def _extract_keywords(query: str) -> list[str]:
    keywords: list[str] = []
    for phrase in _extract_exact_phrases(query):
        keywords.append(phrase.lower())
    for token in _TOKEN_RE.findall(query.lower()):
        if re.fullmatch(r"20\d{2}年?", token):
            continue
        if token in _STOPWORDS:
            continue
        keywords.append(token)
        if re.search(r"[\u4e00-\u9fff]", token) and len(token) > 2:
            keywords.extend(token[index : index + 2] for index in range(len(token) - 1))
    return list(dict.fromkeys(keywords))


def _expand_terms(query: str, keywords: list[str], exact_phrases: list[str]) -> list[str]:
    terms: list[str] = []
    for phrase in exact_phrases:
        terms.append(phrase.lower())
    for keyword in keywords:
        lowered = keyword.lower()
        terms.append(lowered)
        for base, synonyms in _SYNONYM_MAP.items():
            if base in lowered:
                terms.extend(item.lower() for item in synonyms)
    lowered_query = query.lower()
    for pattern, expansions in _PHRASE_EXPANSIONS:
        if pattern.search(lowered_query):
            terms.extend(item.lower() for item in expansions)
    return list(dict.fromkeys(term for term in terms if term))
