from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..runtime import (
    CandidateItem,
    build_group_memberships,
    candidate_adjusted_score,
    future_quota_feasible,
    is_filtered,
    normalize_candidates,
    prefix_is_valid,
    ranking_violations,
    sort_candidates_by_tie_break,
)


@dataclass
class SolverResult:
    ranking: List[CandidateItem]
    score: float
    feasible: bool
    violations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class BaseSolver:
    name = "base"

    def solve(self, dsl: Dict[str, Any], candidates: Iterable[Dict[str, Any]]) -> SolverResult:
        normalized_candidates = normalize_candidates(candidates)
        memberships = build_group_memberships(dsl, normalized_candidates)
        usable_candidates = [candidate for candidate in normalized_candidates if not is_filtered(candidate, dsl, memberships)]
        sorted_candidates = sort_candidates_by_tie_break(usable_candidates, dsl, memberships)
        result = self._solve(dsl, sorted_candidates, memberships)
        result.violations = ranking_violations(result.ranking, dsl, memberships)
        result.feasible = not result.violations and len(result.ranking) == dsl["meta"]["top_k"]
        result.metadata.setdefault("solver", self.name)
        return result

    def _solve(
        self,
        dsl: Dict[str, Any],
        candidates: Sequence[CandidateItem],
        memberships: Dict[str, set[str]],
    ) -> SolverResult:
        raise NotImplementedError


def score_ranking(
    ranking: Sequence[CandidateItem],
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
) -> float:
    return sum(candidate_adjusted_score(candidate, dsl, memberships) for candidate in ranking)


def can_extend_prefix(
    prefix: Sequence[CandidateItem],
    remaining: Sequence[CandidateItem],
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
    top_k: int,
) -> bool:
    if not prefix_is_valid(prefix, dsl):
        return False
    return future_quota_feasible(prefix, remaining, dsl, memberships, top_k)


def exists_feasible_completion(
    prefix: Sequence[CandidateItem],
    remaining: Sequence[CandidateItem],
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
    top_k: int,
) -> bool:
    if not can_extend_prefix(prefix, remaining, dsl, memberships, top_k):
        return False
    if len(prefix) == top_k:
        return True

    candidate_lookup = {candidate.item_id: candidate for candidate in remaining}

    @lru_cache(maxsize=None)
    def search(prefix_ids: tuple[str, ...], remaining_ids: tuple[str, ...]) -> bool:
        current_prefix = list(prefix) + [candidate_lookup[item_id] for item_id in prefix_ids]
        current_remaining = [candidate_lookup[item_id] for item_id in remaining_ids]
        if not can_extend_prefix(current_prefix, current_remaining, dsl, memberships, top_k):
            return False
        if len(current_prefix) == top_k:
            return True

        for index, item_id in enumerate(remaining_ids):
            next_prefix_ids = prefix_ids + (item_id,)
            next_remaining_ids = remaining_ids[:index] + remaining_ids[index + 1 :]
            if search(next_prefix_ids, next_remaining_ids):
                return True
        return False

    return search(tuple(), tuple(candidate.item_id for candidate in remaining))
