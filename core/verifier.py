from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .dsl_parser import parse_ranking_dsl
from .runtime import (
    RankDSLError,
    build_group_memberships,
    future_quota_feasible,
    is_filtered,
    normalize_candidates,
)


@dataclass
class VerificationResult:
    ok: bool
    dsl: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    memberships: Dict[str, set[str]] = field(default_factory=dict)
    filtered_candidate_count: int = 0


def verify_dsl(payload: str | Dict[str, Any], candidates: Optional[Iterable[Dict[str, Any]]] = None) -> VerificationResult:
    try:
        dsl = parse_ranking_dsl(payload)
    except RankDSLError as exc:
        return VerificationResult(ok=False, errors=[exc.to_dict()])

    result = VerificationResult(ok=True, dsl=dsl)
    if candidates is None:
        return result

    normalized_candidates = normalize_candidates(candidates)
    memberships = build_group_memberships(dsl, normalized_candidates)
    result.memberships = memberships

    available_candidates = [candidate for candidate in normalized_candidates if not is_filtered(candidate, dsl, memberships)]
    result.filtered_candidate_count = len(available_candidates)
    top_k = dsl["meta"]["top_k"]
    if len(available_candidates) < top_k:
        result.ok = False
        result.errors.append(
            RankDSLError(
                "infeasible_on_candidates",
                "Not enough candidates remain after filter constraints",
                {"available": len(available_candidates), "top_k": top_k},
            ).to_dict()
        )
        return result

    if not future_quota_feasible([], available_candidates, dsl, memberships, top_k):
        result.ok = False
        result.errors.append(
            RankDSLError(
                "infeasible_on_candidates",
                "Quota constraints cannot be satisfied on the provided candidate set",
            ).to_dict()
        )
        return result

    return result

