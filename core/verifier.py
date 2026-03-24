from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from .dsl_parser import parse_ranking_dsl
from .runtime import (
    RankDSLError,
    build_group_memberships,
    future_quota_feasible,
    is_filtered,
    normalize_candidates,
)


class VerificationError(BaseModel):
    code: str
    error_type: str
    message: str
    suggestion: str
    offending_field: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class VerificationResult:
    ok: bool
    dsl: Optional[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    memberships: Dict[str, set[str]] = field(default_factory=dict)
    filtered_candidate_count: int = 0


def _suggestion_for_error(exc: RankDSLError) -> str:
    if exc.code == "schema_error":
        return "Return one JSON object that matches the RankDSL schema exactly. Do not add extra keys or prose."
    if exc.code == "unknown_field":
        return "Use only supported candidate fields such as genre, dominant_genre, release_year, categories, dominant_category, price, and brand."
    if exc.code == "unsupported_operator":
        return "Use supported operators only. For genre/categories, use == or !=. Keep tie_break fixed."
    if exc.code == "invalid_topk":
        return "Set meta.top_k to a positive integer."
    if exc.code == "infeasible_on_candidates":
        return "Relax filters or lower quota requirements so the candidate set can still fill top_k."
    return "Return one corrected RankDSL JSON object that preserves the intent but satisfies the schema and candidate constraints."


def _classify_error(exc: RankDSLError) -> str:
    if exc.code in {"infeasible_on_candidates"}:
        return "feasibility"
    if exc.code in {"unknown_field"}:
        return "metadata"
    return "schema"


def _to_verification_error(exc: RankDSLError) -> Dict[str, Any]:
    field = exc.details.get("field") or exc.details.get("attribute") or exc.details.get("location")
    return VerificationError(
        code=exc.code,
        error_type=_classify_error(exc),
        message=exc.message,
        suggestion=_suggestion_for_error(exc),
        offending_field=str(field) if field not in (None, "") else None,
        details=exc.details,
    ).model_dump()


def verify_dsl(payload: str | Dict[str, Any], candidates: Optional[Iterable[Dict[str, Any]]] = None) -> VerificationResult:
    try:
        dsl = parse_ranking_dsl(payload)
    except RankDSLError as exc:
        return VerificationResult(ok=False, errors=[_to_verification_error(exc)])

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
        exc = RankDSLError(
            "infeasible_on_candidates",
            "Not enough candidates remain after filter constraints",
            {"available": len(available_candidates), "top_k": top_k},
        )
        result.ok = False
        result.errors.append(_to_verification_error(exc))
        return result

    available_ids = {candidate.item_id for candidate in available_candidates}
    for quota in dsl["constraints"]["quotas"]:
        available_count = len(memberships[quota["target_group"]] & available_ids)
        min_count = quota["min_count"]
        if min_count > available_count:
            exc = RankDSLError(
                "infeasible_on_candidates",
                f"Quota for group {quota['target_group']} requires min_count={min_count}, but only {available_count} candidates are available",
                {
                    "target_group": quota["target_group"],
                    "min_count": min_count,
                    "available": available_count,
                },
            )
            result.ok = False
            result.errors.append(_to_verification_error(exc))
            return result

    if not future_quota_feasible([], available_candidates, dsl, memberships, top_k):
        exc = RankDSLError(
            "infeasible_on_candidates",
            "Quota constraints cannot be satisfied on the provided candidate set",
            {"top_k": top_k},
        )
        result.ok = False
        result.errors.append(_to_verification_error(exc))
        return result

    return result
