from __future__ import annotations

from typing import Any, Dict, List

from ..core.runtime import CandidateItem, normalize_candidates
from ..llm.client import RankDSLLLMClient


def score_adjust_baseline(request: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    adjusted = []
    text = request["constraint_text"].lower()
    dataset_name = request.get("dataset_name", "ml-1m")
    for candidate in candidates:
        score = float(candidate["base_score"])
        genres = set(candidate["genre"])
        categories = set(candidate.get("categories", candidate["genre"]))
        price = candidate.get("price")

        if dataset_name == "ml-1m":
            if "exclude horror" in text and "Horror" in genres:
                continue
            if "3 comedy" in text and "Comedy" in genres:
                score += 1e6
            if "2 children's" in text and "Children's" in genres:
                score += 1e6
        elif dataset_name == "amazon-books":
            if "priced above $20" in text and price not in (None, "") and float(price) > 20:
                continue
            if "3 mystery" in text and "Mystery" in categories:
                score += 1e6
            if "2 science fiction" in text and "Science Fiction" in categories:
                score += 1e6
        adjusted_candidate = dict(candidate)
        adjusted_candidate["base_score"] = score
        adjusted.append(adjusted_candidate)
    adjusted.sort(key=lambda item: (-float(item["base_score"]), item["item_id"]))
    return adjusted[:10]


def parse_direct_rerank_ids(client: RankDSLLLMClient, text: str) -> List[str]:
    payload = client.parse_json_response(text)
    if not isinstance(payload, list):
        raise ValueError("Direct rerank response must be a JSON array")
    return [str(item_id) for item_id in payload[:10]]


def align_candidate_order(candidate_ids: List[str], candidates: List[Dict[str, Any]]) -> List[CandidateItem]:
    lookup = {candidate["item_id"]: candidate for candidate in candidates}
    ordered = [lookup[item_id] for item_id in candidate_ids if item_id in lookup][:10]
    return normalize_candidates(ordered)
