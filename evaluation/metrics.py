from __future__ import annotations

import math
from typing import Dict, Iterable, List

from ..core.runtime import CandidateItem


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

