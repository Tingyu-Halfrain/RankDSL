from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

from ..data.reader_factory import get_reader
from .io import load_jsonl, write_jsonl


def hydrate_candidate_metadata(reader, candidate_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hydrated = []
    for row in candidate_rows:
        meta = reader.get_item_metadata(row["item_id"])
        hydrated.append(
            {
                "item_id": meta["item_id"],
                "title": meta["title"],
                "genre": meta["genre"],
                "categories": meta.get("categories", meta["genre"]),
                "dominant_genre": meta["dominant_genre"],
                "dominant_category": meta.get("dominant_category", meta["dominant_genre"]),
                "release_year": meta["release_year"],
                "price": meta.get("price"),
                "brand": meta.get("brand"),
                "base_score": float(row["base_score"]),
            }
        )
    return hydrated


def ensure_candidate_metadata(
    dataset_dir: str | Path,
    candidate_path: str | Path,
    semantic_cache_path: str | Path | None = None,
) -> None:
    rows = load_jsonl(candidate_path)
    if not rows:
        return

    sample_candidates = rows[0].get("candidates", [])
    if sample_candidates and all("title" in candidate and "genre" in candidate for candidate in sample_candidates):
        return

    reader = get_reader(dataset_dir, semantic_cache_path=semantic_cache_path)
    hydrated_rows = []
    for row in rows:
        hydrated_rows.append(
            {
                "user_id": row["user_id"],
                "target_item_id": row.get("target_item_id", reader.test_target_map.get(row["user_id"])),
                "candidates": hydrate_candidate_metadata(reader, row.get("candidates", [])),
            }
        )
    write_jsonl(candidate_path, hydrated_rows)


def build_popularity_candidates(
    dataset_dir: str | Path,
    output_path: str | Path,
    top_n: int = 20,
    semantic_cache_path: str | Path | None = None,
) -> None:
    reader = get_reader(dataset_dir, semantic_cache_path=semantic_cache_path)
    popularity = Counter()
    for history in reader.filtered_history_map.values():
        popularity.update(history)

    ranked_item_ids = [item_id for item_id, _ in popularity.most_common()]
    rows: List[Dict[str, Any]] = []
    for user_id in reader.eligible_users():
        seen = {item_id for item_id, _, _ in reader.get_user_history(user_id)}
        candidates = []
        for item_id in ranked_item_ids:
            if item_id in seen:
                continue
            candidates.append({"item_id": item_id, "base_score": float(popularity[item_id])})
            if len(candidates) == top_n:
                break
        rows.append(
            {
                "user_id": user_id,
                "target_item_id": reader.test_target_map[user_id],
                "candidates": hydrate_candidate_metadata(reader, candidates),
            }
        )
    write_jsonl(output_path, rows)
