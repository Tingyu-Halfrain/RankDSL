from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ALLOWED_FILTER_FIELDS = {
    "genre",
    "dominant_genre",
    "release_year",
    "categories",
    "dominant_category",
    "price",
    "brand",
}
ALLOWED_DIVERSITY_ATTRIBUTES = {
    "genre",
    "dominant_genre",
    "release_year",
    "categories",
    "dominant_category",
    "brand",
}


class RankDSLError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass(frozen=True)
class CandidateItem:
    item_id: str
    title: str
    base_score: float
    genre: Tuple[str, ...]
    categories: Tuple[str, ...]
    dominant_genre: str
    dominant_category: str
    release_year: Optional[int]
    price: Optional[float]
    brand: Optional[str]
    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CandidateItem":
        genres = payload.get("genre") or ()
        if isinstance(genres, str):
            genres = tuple(part for part in genres.split() if part)
        else:
            genres = tuple(str(part) for part in genres)

        categories = payload.get("categories") or genres
        if isinstance(categories, str):
            categories = tuple(part for part in categories.split() if part)
        else:
            categories = tuple(str(part) for part in categories)

        release_year = payload.get("release_year")
        if release_year in ("", None):
            normalized_year = None
        else:
            normalized_year = int(float(release_year))

        dominant_genre = payload.get("dominant_genre")
        if not dominant_genre:
            dominant_genre = derive_dominant_genre(genres)

        dominant_category = payload.get("dominant_category")
        if not dominant_category:
            dominant_category = derive_dominant_genre(categories)

        price_value = payload.get("price")
        if price_value in ("", None):
            normalized_price = None
        else:
            normalized_price = float(price_value)

        brand_value = payload.get("brand")
        normalized_brand = str(brand_value).strip() if brand_value not in ("", None) else None

        raw = dict(payload)
        raw["genre"] = list(genres)
        raw["categories"] = list(categories)
        raw["dominant_genre"] = dominant_genre
        raw["dominant_category"] = dominant_category
        raw["release_year"] = normalized_year
        raw["price"] = normalized_price
        raw["brand"] = normalized_brand

        return cls(
            item_id=str(payload["item_id"]),
            title=str(payload.get("title") or payload.get("movie_title") or ""),
            base_score=float(payload.get("base_score", 0.0)),
            genre=genres,
            categories=categories,
            dominant_genre=dominant_genre,
            dominant_category=dominant_category,
            release_year=normalized_year,
            price=normalized_price,
            brand=normalized_brand,
            raw=raw,
        )


def derive_dominant_genre(genres: Sequence[str]) -> str:
    return sorted(genres)[0] if genres else "UNKNOWN"


def normalize_candidates(candidates: Iterable[Dict[str, Any]]) -> List[CandidateItem]:
    return [CandidateItem.from_dict(candidate) for candidate in candidates]


def get_candidate_field(candidate: CandidateItem, field: str) -> Any:
    if field == "genre":
        return candidate.genre
    if field == "categories":
        return candidate.categories
    if field == "dominant_genre":
        return candidate.dominant_genre
    if field == "dominant_category":
        return candidate.dominant_category
    if field == "release_year":
        return candidate.release_year
    if field == "price":
        return candidate.price
    if field == "brand":
        return candidate.brand
    raise RankDSLError("unknown_field", f"Unsupported candidate field: {field}", {"field": field})


def eval_atom(atom: Dict[str, Any], candidate: CandidateItem) -> bool:
    field = atom["field"]
    op = atom["op"]
    value = atom["value"]
    candidate_value = get_candidate_field(candidate, field)

    if field in {"genre", "categories"}:
        if op == "==":
            return value in candidate_value
        if op == "!=":
            return value not in candidate_value
        raise RankDSLError(
            "unsupported_operator",
            f"Operator {op} is unsupported for field {field}",
            {"field": field, "operator": op},
        )

    if candidate_value is None:
        return False

    if op == "==":
        return candidate_value == value
    if op == "!=":
        return candidate_value != value
    if op == ">":
        return candidate_value > value
    if op == ">=":
        return candidate_value >= value
    if op == "<":
        return candidate_value < value
    if op == "<=":
        return candidate_value <= value
    raise RankDSLError(
        "unsupported_operator",
        f"Unsupported operator: {op}",
        {"field": field, "operator": op},
    )


def evaluate_filter_ast(ast: Dict[str, Any], candidate: CandidateItem) -> bool:
    node_type = ast["type"]
    if node_type == "atom":
        return eval_atom(ast, candidate)
    if node_type == "and":
        return evaluate_filter_ast(ast["left"], candidate) and evaluate_filter_ast(ast["right"], candidate)
    if node_type == "or":
        return evaluate_filter_ast(ast["left"], candidate) or evaluate_filter_ast(ast["right"], candidate)
    raise RankDSLError("syntax_error", f"Unsupported AST node type: {node_type}", {"node_type": node_type})


def build_group_memberships(dsl: Dict[str, Any], candidates: Sequence[CandidateItem]) -> Dict[str, set[str]]:
    memberships: Dict[str, set[str]] = {}
    for group in dsl["groups"]:
        matched = {candidate.item_id for candidate in candidates if evaluate_filter_ast(group["ast"], candidate)}
        memberships[group["group_id"]] = matched
    return memberships


def candidate_adjusted_score(
    candidate: CandidateItem,
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
) -> float:
    score = dsl["objective"]["base_score_weight"] * candidate.base_score
    for boost in dsl["objective"]["group_boosts"]:
        if candidate.item_id in memberships[boost["target_group"]]:
            score += boost["weight"]
    return score


def is_filtered(candidate: CandidateItem, dsl: Dict[str, Any], memberships: Dict[str, set[str]]) -> bool:
    for filter_rule in dsl["constraints"]["filters"]:
        if filter_rule["action"] == "exclude" and candidate.item_id in memberships[filter_rule["target_group"]]:
            return True
    return False


def quota_counts_for_ranking(
    ranking: Sequence[CandidateItem],
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for quota in dsl["constraints"]["quotas"]:
        counts[quota["target_group"]] = sum(
            1 for candidate in ranking if candidate.item_id in memberships[quota["target_group"]]
        )
    return counts


def diversity_violations(
    ranking: Sequence[CandidateItem],
    dsl: Dict[str, Any],
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for diversity in dsl["constraints"]["diversity"]:
        attribute = diversity["attribute"]
        window_size = diversity["window_size"]
        max_repetition = diversity["max_repetition"]
        for start in range(max(0, len(ranking) - window_size + 1)):
            window = ranking[start : start + window_size]
            values = [get_candidate_field(candidate, attribute) for candidate in window]
            value_counts = Counter(values)
            for value, count in value_counts.items():
                if count > max_repetition:
                    violations.append(
                        {
                            "type": "diversity",
                            "attribute": attribute,
                            "window_start": start,
                            "window_size": window_size,
                            "value": value,
                            "count": count,
                            "max_repetition": max_repetition,
                        }
                    )
    return violations


def ranking_violations(
    ranking: Sequence[CandidateItem],
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    ranking_ids = {candidate.item_id for candidate in ranking}

    for filter_rule in dsl["constraints"]["filters"]:
        filtered_hits = ranking_ids.intersection(memberships[filter_rule["target_group"]])
        if filtered_hits:
            violations.append(
                {
                    "type": "filter",
                    "target_group": filter_rule["target_group"],
                    "action": filter_rule["action"],
                    "matched_items": sorted(filtered_hits),
                }
            )

    for quota in dsl["constraints"]["quotas"]:
        count = sum(1 for candidate in ranking if candidate.item_id in memberships[quota["target_group"]])
        min_count = quota["min_count"]
        max_count = quota["max_count"]
        if count < min_count or (max_count is not None and count > max_count):
            violations.append(
                {
                    "type": "quota",
                    "target_group": quota["target_group"],
                    "count": count,
                    "min_count": min_count,
                    "max_count": max_count,
                }
            )

    violations.extend(diversity_violations(ranking, dsl))
    return violations


def ranking_genre_coverage(ranking: Sequence[CandidateItem]) -> int:
    covered = set()
    for candidate in ranking:
        if candidate.genre:
            covered.update(candidate.genre)
        else:
            covered.update(candidate.categories)
    return len(covered)


def sort_candidates_by_tie_break(
    candidates: Iterable[CandidateItem],
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
) -> List[CandidateItem]:
    return sorted(
        candidates,
        key=lambda candidate: (-candidate_adjusted_score(candidate, dsl, memberships), candidate.item_id),
    )


def prefix_is_valid(
    prefix: Sequence[CandidateItem],
    dsl: Dict[str, Any],
) -> bool:
    return not diversity_violations(prefix, dsl)


def quota_count_for_group(
    ranking: Sequence[CandidateItem],
    target_group: str,
    memberships: Dict[str, set[str]],
) -> int:
    return sum(1 for candidate in ranking if candidate.item_id in memberships[target_group])


def future_quota_feasible(
    prefix: Sequence[CandidateItem],
    remaining: Sequence[CandidateItem],
    dsl: Dict[str, Any],
    memberships: Dict[str, set[str]],
    top_k: int,
) -> bool:
    slots_left = top_k - len(prefix)
    if slots_left < 0:
        return False

    remaining_ids = {candidate.item_id for candidate in remaining}
    for quota in dsl["constraints"]["quotas"]:
        current = quota_count_for_group(prefix, quota["target_group"], memberships)
        if quota["max_count"] is not None and current > quota["max_count"]:
            return False
        available = len(remaining_ids.intersection(memberships[quota["target_group"]]))
        if current + min(slots_left, available) < quota["min_count"]:
            return False
    return True
