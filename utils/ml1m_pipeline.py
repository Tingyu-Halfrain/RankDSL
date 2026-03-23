from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..core.runtime import CandidateItem
from ..experiments.candidate_builder import build_popularity_candidates
from ..experiments.io import load_candidate_lookup, load_jsonl
from ..experiments.request_builder import export_requests, generate_requests
from ..experiments.scenarios import SCENARIOS


def top10_item_dicts(ranking: Sequence[CandidateItem]) -> List[Dict[str, Any]]:
    return [
        {
            "item_id": candidate.item_id,
            "title": candidate.title,
            "genre": list(candidate.genre),
            "dominant_genre": candidate.dominant_genre,
            "release_year": candidate.release_year,
            "base_score": candidate.base_score,
        }
        for candidate in ranking[:10]
    ]
