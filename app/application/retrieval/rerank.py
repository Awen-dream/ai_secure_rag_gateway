from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

from app.application.retrieval.planning import RecallPlan
from app.domain.retrieval.backends import BackendSearchHit
from app.domain.retrieval.models import RetrievalProfile, RetrievalResult
from app.domain.retrieval.rerankers import RetrievalReranker, sort_by_score, weighted_fusion


@dataclass(frozen=True)
class RerankExecutionResult:
    pre_rerank_results: list[RetrievalResult]
    results: list[RetrievalResult]


class RetrievalRerankService:
    """Prepare rerank candidates from multi-backend hits and run the configured reranker."""

    def __init__(self, reranker: RetrievalReranker | None = None) -> None:
        self.reranker = reranker

    def build_rerank_candidates(
        self,
        keyword_hits: list[BackendSearchHit],
        vector_hits: list[BackendSearchHit],
        recall_plan: RecallPlan,
    ) -> list[RetrievalResult]:
        profile = recall_plan.profile
        rewrite_plan = recall_plan.query_plan.rewrite_plan
        merged: dict[str, RetrievalResult] = {}
        max_keyword = max((hit.score for hit in keyword_hits), default=0.0)
        max_vector = max((hit.score for hit in vector_hits), default=0.0)
        keyword_rank = {
            hit.chunk.id: index + 1
            for index, hit in enumerate(sorted(keyword_hits, key=lambda item: item.score, reverse=True))
        }
        vector_rank = {
            hit.chunk.id: index + 1
            for index, hit in enumerate(sorted(vector_hits, key=lambda item: item.score, reverse=True))
        }

        for hit in keyword_hits:
            merged[hit.chunk.id] = RetrievalResult(
                document=hit.document,
                chunk=hit.chunk,
                score=0.0,
                keyword_score=hit.score,
                vector_score=0.0,
                matched_terms=hit.matched_terms,
                retrieval_sources=[hit.backend],
            )

        for hit in vector_hits:
            if hit.chunk.id not in merged:
                merged[hit.chunk.id] = RetrievalResult(
                    document=hit.document,
                    chunk=hit.chunk,
                    score=0.0,
                    keyword_score=0.0,
                    vector_score=hit.score,
                    matched_terms=[],
                    retrieval_sources=[hit.backend],
                )
                continue

            existing = merged[hit.chunk.id]
            existing.vector_score = hit.score
            if hit.backend not in existing.retrieval_sources:
                existing.retrieval_sources.append(hit.backend)

        normalized_results: list[RetrievalResult] = []
        for result in merged.values():
            keyword_normalized = result.keyword_score / max_keyword if max_keyword else 0.0
            vector_normalized = result.vector_score / max_vector if max_vector else 0.0
            title_lower = result.document.title.lower()
            section_lower = result.chunk.section_name.lower()
            body_lower = result.chunk.text.lower()
            title_boost = (
                profile.title_boost
                if any(term in title_lower for term in rewrite_plan.expanded_terms or rewrite_plan.keywords)
                else 0.0
            )
            phrase_boost = sum(
                0.04 for phrase in rewrite_plan.exact_phrases[:2] if phrase.lower() in body_lower
            )
            tag_boost = (
                0.05
                if rewrite_plan.tag_filters
                and any(tag.lower() in {item.lower() for item in result.document.tags} for tag in rewrite_plan.tag_filters)
                else 0.0
            )
            exact_match_boost = min(
                sum(0.03 for term in recall_plan.exact_match_terms if term in f"{title_lower} {section_lower} {body_lower}"),
                0.12,
            )
            year_boost = (
                0.04
                if recall_plan.filters.year_filters
                and (
                    result.document.updated_at.year in recall_plan.filters.year_filters
                    or result.document.created_at.year in recall_plan.filters.year_filters
                )
                else 0.0
            )
            recency_boost = self._recency_boost(result.document.updated_at, rewrite_plan.recency_hint)
            rank_fusion = self._reciprocal_rank_fusion(
                keyword_rank.get(result.chunk.id),
                vector_rank.get(result.chunk.id),
                profile,
            )
            result.score = weighted_fusion(
                keyword_score=keyword_normalized,
                vector_score=vector_normalized,
                keyword_weight=profile.keyword_weight,
                vector_weight=profile.vector_weight,
                title_boost=title_boost,
            )
            result.score = round(
                result.score + phrase_boost + tag_boost + exact_match_boost + year_boost + recency_boost + rank_fusion,
                4,
            )
            result.keyword_score = round(keyword_normalized, 4)
            result.vector_score = round(vector_normalized, 4)
            normalized_results.append(result)
        return sort_by_score(normalized_results)

    def rerank_results(self, results: list[RetrievalResult], recall_plan: RecallPlan) -> list[RetrievalResult]:
        return self.execute_rerank(results, recall_plan).results

    def execute_rerank(self, results: list[RetrievalResult], recall_plan: RecallPlan) -> RerankExecutionResult:
        if not results:
            return RerankExecutionResult(pre_rerank_results=[], results=[])

        sorted_results = sort_by_score(results)
        trace_by_chunk_id = {item.chunk.id: item.model_copy(deep=True) for item in sorted_results}
        for trace in trace_by_chunk_id.values():
            trace.selection_status = "candidate"

        rewrite_plan = recall_plan.query_plan.rewrite_plan
        top_score = sorted_results[0].score
        filtered_candidates = [
            item
            for item in sorted_results
            if item.score >= recall_plan.profile.min_score
            and item.score >= top_score * recall_plan.profile.relative_score_cutoff
        ]
        filtered_ids = {item.chunk.id for item in filtered_candidates}
        for item in sorted_results:
            trace = trace_by_chunk_id[item.chunk.id]
            reasons: list[str] = []
            if item.score < recall_plan.profile.min_score:
                reasons.append("below_min_score")
            if item.score < top_score * recall_plan.profile.relative_score_cutoff:
                reasons.append("below_relative_score_cutoff")
            if reasons:
                trace.selection_status = "dropped"
                trace.selection_reasons = reasons
            else:
                trace.selection_status = "passed_threshold"
        if self.reranker:
            filtered_candidates = self.reranker.rerank(rewrite_plan.rewritten_query, filtered_candidates)
        reranked_ids = {item.chunk.id for item in filtered_candidates}
        for chunk_id in filtered_ids - reranked_ids:
            trace = trace_by_chunk_id[chunk_id]
            trace.selection_status = "dropped"
            trace.selection_reasons = list(dict.fromkeys([*trace.selection_reasons, "dropped_by_reranker"]))

        diversity_ordered, diversity_overflow_ids = self._apply_document_diversity(
            filtered_candidates,
            recall_plan.max_chunks_per_document,
        )
        for chunk_id in diversity_overflow_ids:
            trace = trace_by_chunk_id[chunk_id]
            trace.selection_reasons = list(dict.fromkeys([*trace.selection_reasons, "deprioritized_by_doc_diversity"]))

        final_results = diversity_ordered[: recall_plan.candidate_pool]
        final_ids = {item.chunk.id for item in final_results}
        for item in final_results:
            trace = trace_by_chunk_id[item.chunk.id]
            trace.selection_status = "selected"
            trace.selection_reasons = list(dict.fromkeys([*trace.selection_reasons, "selected_after_rerank"]))
            item.selection_status = trace.selection_status
            item.selection_reasons = list(trace.selection_reasons)
        for chunk_id in reranked_ids - final_ids:
            trace = trace_by_chunk_id[chunk_id]
            trace.selection_status = "dropped"
            trace.selection_reasons = list(dict.fromkeys([*trace.selection_reasons, "pruned_by_candidate_pool"]))
        return RerankExecutionResult(
            pre_rerank_results=[trace_by_chunk_id[item.chunk.id] for item in sorted_results],
            results=final_results,
        )

    @staticmethod
    def _apply_document_diversity(
        results: list[RetrievalResult],
        max_chunks_per_document: int,
    ) -> tuple[list[RetrievalResult], set[str]]:
        if max_chunks_per_document <= 0:
            return results, set()

        kept: list[RetrievalResult] = []
        overflow: list[RetrievalResult] = []
        overflow_ids: set[str] = set()
        doc_counts: dict[str, int] = defaultdict(int)
        for result in results:
            if doc_counts[result.document.id] < max_chunks_per_document:
                kept.append(result)
                doc_counts[result.document.id] += 1
            else:
                overflow.append(result)
                overflow_ids.add(result.chunk.id)
        return kept + overflow, overflow_ids

    @staticmethod
    def _reciprocal_rank_fusion(
        keyword_rank: int | None,
        vector_rank: int | None,
        profile: RetrievalProfile,
        k: int = 60,
    ) -> float:
        score = 0.0
        if keyword_rank is not None:
            score += (1.0 / (k + keyword_rank)) * (0.45 + profile.keyword_weight)
        if vector_rank is not None:
            score += (1.0 / (k + vector_rank)) * (0.45 + profile.vector_weight)
        return round(score, 4)

    @staticmethod
    def _recency_boost(updated_at: datetime, recency_hint: bool) -> float:
        if not recency_hint:
            return 0.0
        age_days = max((datetime.utcnow() - updated_at).days, 0)
        if age_days <= 30:
            return 0.06
        if age_days <= 180:
            return 0.03
        if age_days <= 365:
            return 0.015
        return 0.0
