from __future__ import annotations

from typing import Dict, List, Sequence

from .base_solver import BaseSolver, SolverResult, exists_feasible_completion, score_ranking
from ..runtime import CandidateItem, candidate_adjusted_score


class GreedySolver(BaseSolver):
    name = "greedy"

    def _solve(
        self,
        dsl: Dict[str, any],
        candidates: Sequence[CandidateItem],
        memberships: Dict[str, set[str]],
    ) -> SolverResult:
        ranking: List[CandidateItem] = []
        remaining = list(candidates)
        top_k = dsl["meta"]["top_k"]

        while len(ranking) < top_k and remaining:
            chosen = None
            for candidate in remaining:
                candidate_prefix = ranking + [candidate]
                candidate_remaining = [item for item in remaining if item.item_id != candidate.item_id]
                if exists_feasible_completion(candidate_prefix, candidate_remaining, dsl, memberships, top_k):
                    chosen = candidate
                    break

            if chosen is None:
                break

            ranking.append(chosen)
            remaining = [item for item in remaining if item.item_id != chosen.item_id]

        return SolverResult(
            ranking=ranking,
            score=score_ranking(ranking, dsl, memberships),
            feasible=False,
            violations=[],
            metadata={
                "selected_scores": [
                    candidate_adjusted_score(candidate, dsl, memberships) for candidate in ranking
                ]
            },
        )
