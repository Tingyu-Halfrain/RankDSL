from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .base_solver import BaseSolver, SolverResult, can_extend_prefix, score_ranking
from ..runtime import CandidateItem, candidate_adjusted_score


class ILPSolver(BaseSolver):
    name = "ilp_fallback_exact"

    def _solve(
        self,
        dsl: Dict[str, any],
        candidates: Sequence[CandidateItem],
        memberships: Dict[str, set[str]],
    ) -> SolverResult:
        top_k = dsl["meta"]["top_k"]
        best_ranking: List[CandidateItem] = []
        best_score = float("-inf")
        seen_best: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float] = {}

        def upper_bound(prefix_score: float, remaining: Sequence[CandidateItem], slots_left: int) -> float:
            extra = sum(
                candidate_adjusted_score(candidate, dsl, memberships)
                for candidate in remaining[:slots_left]
            )
            return prefix_score + extra

        def dfs(prefix: List[CandidateItem], remaining: List[CandidateItem], prefix_score: float) -> None:
            nonlocal best_ranking, best_score

            state_key = (
                tuple(candidate.item_id for candidate in prefix[-2:]),
                tuple(candidate.item_id for candidate in remaining),
            )
            cached = seen_best.get(state_key)
            if cached is not None and cached >= prefix_score:
                return
            seen_best[state_key] = prefix_score

            if len(prefix) == top_k:
                if prefix_score > best_score:
                    best_ranking = list(prefix)
                    best_score = prefix_score
                return

            slots_left = top_k - len(prefix)
            if len(remaining) < slots_left:
                return
            if upper_bound(prefix_score, remaining, slots_left) <= best_score:
                return
            if not can_extend_prefix(prefix, remaining, dsl, memberships, top_k):
                return

            for index, candidate in enumerate(remaining):
                next_prefix = prefix + [candidate]
                next_remaining = remaining[:index] + remaining[index + 1 :]
                if not can_extend_prefix(next_prefix, next_remaining, dsl, memberships, top_k):
                    continue
                dfs(
                    next_prefix,
                    next_remaining,
                    prefix_score + candidate_adjusted_score(candidate, dsl, memberships),
                )

        dfs([], list(candidates), 0.0)

        return SolverResult(
            ranking=best_ranking,
            score=best_score if best_score != float("-inf") else 0.0,
            feasible=False,
            violations=[],
            metadata={"search_space": len(candidates), "exact": True},
        )
