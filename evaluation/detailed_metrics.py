from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence

from ..core.dsl_parser import parse_ranking_dsl
from ..core.runtime import (
    CandidateItem,
    build_group_memberships,
    get_candidate_field,
    normalize_candidates,
)


def _normalize_ranking(ranking: Sequence[Dict[str, Any]] | Sequence[CandidateItem]) -> List[CandidateItem]:
    if not ranking:
        return []
    first = ranking[0]
    if isinstance(first, CandidateItem):
        return list(ranking)  # type: ignore[arg-type]
    return normalize_candidates(ranking)  # type: ignore[arg-type]


def _normalize_reference_dsl(reference_dsl: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(reference_dsl, dict):
        groups = reference_dsl.get("groups", [])
        if groups and isinstance(groups[0], dict) and "ast" in groups[0]:
            return reference_dsl
    return parse_ranking_dsl(reference_dsl)


def _candidate_tokens(candidate: CandidateItem) -> set[str]:
    tokens = set(candidate.genre) or set(candidate.categories)
    if not tokens and candidate.dominant_genre:
        tokens.add(candidate.dominant_genre)
    return {str(token) for token in tokens if str(token)}


def _jaccard_similarity(left: CandidateItem, right: CandidateItem) -> float:
    left_tokens = _candidate_tokens(left)
    right_tokens = _candidate_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def ild_score(ranking: Sequence[Dict[str, Any]] | Sequence[CandidateItem]) -> float:
    items = _normalize_ranking(ranking)
    if len(items) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            total += 1.0 - _jaccard_similarity(items[left_index], items[right_index])
            pairs += 1
    return total / pairs if pairs else 0.0


def quota_status(
    reference_dsl: str | Dict[str, Any],
    ranking: Sequence[Dict[str, Any]] | Sequence[CandidateItem],
) -> List[Dict[str, Any]]:
    dsl = _normalize_reference_dsl(reference_dsl)
    items = _normalize_ranking(ranking)
    memberships = build_group_memberships(dsl, items)
    status: List[Dict[str, Any]] = []
    for quota in dsl["constraints"]["quotas"]:
        actual_count = sum(1 for item in items if item.item_id in memberships[quota["target_group"]])
        min_count = quota["min_count"]
        max_count = quota["max_count"]
        satisfied = actual_count >= min_count and (max_count is None or actual_count <= max_count)
        status.append(
            {
                "target_group": quota["target_group"],
                "actual_count": actual_count,
                "required_min": min_count,
                "required_max": max_count,
                "satisfied": satisfied,
            }
        )
    return status


def sliding_window_stats(
    reference_dsl: str | Dict[str, Any],
    ranking: Sequence[Dict[str, Any]] | Sequence[CandidateItem],
) -> Dict[str, Any]:
    dsl = _normalize_reference_dsl(reference_dsl)
    items = _normalize_ranking(ranking)
    rules = dsl["constraints"]["diversity"]
    if not rules:
        return {
            "sliding_window_ok": True,
            "max_rep_in_any_window": 0,
            "sliding_violation_rate": 0.0,
        }

    violating_windows = 0
    total_windows = 0
    max_rep_in_any_window = 0

    for rule in rules:
        attribute = rule["attribute"]
        window_size = rule["window_size"]
        max_repetition = rule["max_repetition"]
        if len(items) < window_size:
            continue
        for start in range(len(items) - window_size + 1):
            total_windows += 1
            window = items[start : start + window_size]
            value_counts: Dict[Any, int] = {}
            for item in window:
                value = get_candidate_field(item, attribute)
                value_counts[value] = value_counts.get(value, 0) + 1
            window_max = max(value_counts.values()) if value_counts else 0
            max_rep_in_any_window = max(max_rep_in_any_window, window_max)
            if window_max > max_repetition:
                violating_windows += 1

    sliding_window_ok = violating_windows == 0
    sliding_violation_rate = violating_windows / total_windows if total_windows else 0.0
    return {
        "sliding_window_ok": sliding_window_ok,
        "max_rep_in_any_window": max_rep_in_any_window,
        "sliding_violation_rate": sliding_violation_rate,
    }


def detailed_constraint_status(
    reference_dsl: str | Dict[str, Any],
    ranking: Sequence[Dict[str, Any]] | Sequence[CandidateItem],
) -> Dict[str, Any]:
    dsl = _normalize_reference_dsl(reference_dsl)
    items = _normalize_ranking(ranking)
    memberships = build_group_memberships(dsl, items)
    ranking_ids = {item.item_id for item in items}

    has_filters = bool(dsl["constraints"]["filters"])
    has_quotas = bool(dsl["constraints"]["quotas"])
    has_diversity = bool(dsl["constraints"]["diversity"])

    filter_ok = True
    for filter_rule in dsl["constraints"]["filters"]:
        if ranking_ids.intersection(memberships[filter_rule["target_group"]]):
            filter_ok = False
            break

    quotas = quota_status(dsl, items)
    quotas_ok = all(quota["satisfied"] for quota in quotas) if quotas else True

    sliding = sliding_window_stats(dsl, items)
    diversity_ok = sliding["sliding_window_ok"]

    return {
        "has_filter_constraints": has_filters,
        "has_quota_constraints": has_quotas,
        "has_diversity_constraints": has_diversity,
        "filter_ok": filter_ok,
        "quota_status": quotas,
        "quota_satisfaction": float(quotas_ok),
        "diversity_satisfaction": float(diversity_ok),
        "sliding_window_ok": sliding["sliding_window_ok"],
        "max_rep_in_any_window": sliding["max_rep_in_any_window"],
        "sliding_violation_rate": sliding["sliding_violation_rate"],
        "ild_score": ild_score(items),
    }


def enrich_request_result(result: Dict[str, Any]) -> Dict[str, Any]:
    enriched = deepcopy(result)
    reference_dsl = enriched.get("reference_dsl")
    if not reference_dsl:
        return enriched

    def attach(record: Dict[str, Any]) -> Dict[str, Any]:
        ranking = record.get("ranking")
        if not ranking:
            return record
        record.update(detailed_constraint_status(reference_dsl, ranking))
        return record

    if "base_recall" in enriched:
        attach(enriched["base_recall"])
    if "score_adjust_greedy" in enriched:
        attach(enriched["score_adjust_greedy"])

    for run in enriched.get("prompt_only_direct_rerank", {}).get("runs", []):
        attach(run)

    for run in enriched.get("rankdsl", {}).get("runs", []):
        if "greedy" in run:
            attach(run["greedy"])
        if "ilp" in run:
            attach(run["ilp"])

    return enriched
