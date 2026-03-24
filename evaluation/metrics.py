from __future__ import annotations

import math
from typing import Dict, Iterable, List

from ..core.runtime import CandidateItem
from .detailed_metrics import enrich_request_result


def hit_at_10(ranking: List[CandidateItem], target_item_id: str) -> float:
    return float(any(candidate.item_id == target_item_id for candidate in ranking[:10]))


def ndcg_at_10(ranking: List[CandidateItem], target_item_id: str) -> float:
    for index, candidate in enumerate(ranking[:10]):
        if candidate.item_id == target_item_id:
            return 1.0 / math.log2(index + 2)
    return 0.0


def aggregate_method(results: List[Dict[str, object]], accessor) -> Dict[str, float]:
    metrics = {
        "hit@10": [],
        "ndcg@10": [],
        "constraint_satisfaction": [],
        "violation_count": [],
        "genre_coverage": [],
        "latency": [],
        "filter_ok": [],
        "quota_satisfaction": [],
        "diversity_satisfaction": [],
        "sliding_window_ok": [],
        "sliding_violation_rate": [],
        "ild_score": [],
    }
    for result in results:
        records = accessor(result)
        if records is None:
            continue
        if isinstance(records, dict):
            records = [records]
        for record in records:
            for key in metrics:
                if key in record:
                    metrics[key].append(float(record[key]))
    return {key: (sum(values) / len(values) if values else 0.0) for key, values in metrics.items()}


def _average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def calculate_summary(results: List[Dict[str, object]]) -> Dict[str, float | Dict[str, float]]:
    enriched_results = [enrich_request_result(result) for result in results]
    summary: Dict[str, float | Dict[str, float]] = {
        "base_recall": aggregate_method(enriched_results, lambda row: row["base_recall"]),
        "score_adjust_greedy": aggregate_method(enriched_results, lambda row: row["score_adjust_greedy"]),
        "prompt_only_direct_rerank": aggregate_method(enriched_results, lambda row: row["prompt_only_direct_rerank"]["runs"]),
        "rankdsl_greedy": aggregate_method(enriched_results, lambda row: [run["greedy"] for run in row["rankdsl"]["runs"]]),
        "rankdsl_ilp": aggregate_method(enriched_results, lambda row: [run["ilp"] for run in row["rankdsl"]["runs"]]),
        "compile_success_rate": _average([float(row["rankdsl"]["compile_success_rate"]) for row in enriched_results]),
        "canonical_program_agreement": _average([float(row["rankdsl"]["canonical_program_agreement"]) for row in enriched_results]),
        "rankdsl_constraint_satisfaction_variance": _average(
            [float(row["rankdsl"]["constraint_satisfaction_variance"]) for row in enriched_results]
        ),
        "prompt_only_constraint_satisfaction_variance": _average(
            [float(row["prompt_only_direct_rerank"]["constraint_satisfaction_variance"]) for row in enriched_results]
        ),
    }

    filter_scores: List[float] = []
    quota_scores: List[float] = []
    diversity_scores: List[float] = []
    sliding_ok_scores: List[float] = []
    ild_scores: List[float] = []
    successful_ndcgs: List[float] = []

    for row in enriched_results:
        for run in row["rankdsl"]["runs"]:
            ilp = run.get("ilp", {})
            if "compile_error" not in run:
                if "ndcg@10" in ilp:
                    successful_ndcgs.append(float(ilp["ndcg@10"]))
            if ilp.get("has_filter_constraints"):
                filter_scores.append(float(ilp.get("filter_ok", 0.0)))
            if ilp.get("has_quota_constraints"):
                quota_scores.append(float(ilp.get("quota_satisfaction", 0.0)))
            if ilp.get("has_diversity_constraints"):
                diversity_scores.append(float(ilp.get("diversity_satisfaction", 0.0)))
                sliding_ok_scores.append(float(ilp.get("sliding_window_ok", 0.0)))
            if "ild_score" in ilp:
                ild_scores.append(float(ilp["ild_score"]))

    summary["filter_satisfaction"] = _average(filter_scores)
    summary["quota_satisfaction"] = _average(quota_scores)
    summary["diversity_satisfaction"] = _average(diversity_scores)
    summary["sliding_window_ok_rate"] = _average(sliding_ok_scores)
    summary["ild_score_avg"] = _average(ild_scores)
    summary["ndcg_only_successful"] = _average(successful_ndcgs)
    return summary
