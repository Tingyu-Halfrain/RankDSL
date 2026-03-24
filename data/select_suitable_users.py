from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RankDSL.core.runtime import derive_dominant_genre
from RankDSL.core.verifier import verify_dsl
from RankDSL.experiments.dataset_specs import get_dataset_spec
from RankDSL.experiments.io import load_jsonl, write_jsonl
from RankDSL.data.reader_factory import get_reader


ML1M_SCENARIO_TYPE_MAP = {
    "filter_horror": "filter",
    "quota_comedy": "quota",
    "quota_children": "quota",
    "quota_comedy_filter_horror": "quota_filter",
    "diversity_dominant_genre": "diversity",
    "quota_children_diversity_filter_horror": "quota_diversity_filter",
}


def candidate_pool_stats(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    horror_count = 0
    non_horror_count = 0
    children_count = 0
    comedy_count = 0
    dominant_genres = set()

    for candidate in candidates:
        genres = {str(value) for value in candidate.get("genre", [])}
        dominant_genre = str(candidate.get("dominant_genre") or derive_dominant_genre(sorted(genres)))
        dominant_genres.add(dominant_genre)

        if "Horror" in genres:
            horror_count += 1
        else:
            non_horror_count += 1

        if "Children's" in genres:
            children_count += 1
        if "Comedy" in genres:
            comedy_count += 1

    return {
        "candidate_count": len(candidates),
        "horror_count": horror_count,
        "non_horror_count": non_horror_count,
        "children_count": children_count,
        "comedy_count": comedy_count,
        "distinct_dominant_genres": len(dominant_genres),
        "dominant_genres": sorted(dominant_genres),
    }


def _meets_basic_thresholds(
    stats: Dict[str, Any],
    min_horror_candidates: int,
    min_children_candidates: int,
    min_dominant_genres: int,
    min_non_horror_candidates: int,
) -> bool:
    return (
        stats["horror_count"] >= min_horror_candidates
        and stats["children_count"] >= min_children_candidates
        and stats["distinct_dominant_genres"] >= min_dominant_genres
        and stats["non_horror_count"] >= min_non_horror_candidates
    )


def build_ml1m_suitable_requests(
    candidate_rows: Sequence[Dict[str, Any]],
    reader: Any,
    users_per_scenario: int = 100,
    min_horror_candidates: int = 1,
    min_children_candidates: int = 2,
    min_comedy_candidates: int = 3,
    min_dominant_genres: int = 3,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    spec = get_dataset_spec("ml-1m")
    scenario_lookup = {scenario["scenario_id"]: scenario for scenario in spec["scenarios"]}
    reference_builder = spec["reference_builder"]
    eligible_scenarios = [
        scenario["scenario_id"]
        for scenario in spec["scenarios"]
        if scenario["scenario_id"] in ML1M_SCENARIO_TYPE_MAP
    ]
    scenario_type_lookup = {
        scenario_id: ML1M_SCENARIO_TYPE_MAP[scenario_id]
        for scenario_id in eligible_scenarios
    }
    selected_by_scenario: Dict[str, List[Dict[str, Any]]] = {scenario_id: [] for scenario_id in eligible_scenarios}

    def scenario_is_suitable(scenario_id: str, stats: Dict[str, Any], candidates: Sequence[Dict[str, Any]], user_summary: str) -> bool:
        if len(selected_by_scenario[scenario_id]) >= users_per_scenario:
            return False
        if scenario_id == "filter_horror":
            basic_ok = stats["horror_count"] >= min_horror_candidates and stats["non_horror_count"] >= top_k
        elif scenario_id == "quota_comedy":
            basic_ok = stats["comedy_count"] >= min_comedy_candidates
        elif scenario_id == "quota_children":
            basic_ok = stats["children_count"] >= min_children_candidates
        elif scenario_id == "quota_comedy_filter_horror":
            basic_ok = (
                stats["comedy_count"] >= min_comedy_candidates
                and stats["horror_count"] >= min_horror_candidates
                and stats["non_horror_count"] >= top_k
            )
        elif scenario_id == "diversity_dominant_genre":
            basic_ok = stats["distinct_dominant_genres"] >= min_dominant_genres
        elif scenario_id == "quota_children_diversity_filter_horror":
            basic_ok = (
                stats["children_count"] >= min_children_candidates
                and stats["horror_count"] >= min_horror_candidates
                and stats["non_horror_count"] >= top_k
                and stats["distinct_dominant_genres"] >= min_dominant_genres
            )
        else:
            basic_ok = _meets_basic_thresholds(
                stats,
                min_horror_candidates=min_horror_candidates,
                min_children_candidates=min_children_candidates,
                min_dominant_genres=min_dominant_genres,
                min_non_horror_candidates=top_k,
            )
        if not basic_ok:
            return False
        reference_dsl = reference_builder(f"{scenario_id}-probe", user_summary, scenario_id)
        return verify_dsl(reference_dsl, candidates).ok

    for row in sorted(candidate_rows, key=lambda value: str(value["user_id"])):
        if all(len(rows) >= users_per_scenario for rows in selected_by_scenario.values()):
            break

        user_id = str(row["user_id"])
        target_item_id = str(row.get("target_item_id") or reader.test_target_map.get(user_id) or "")
        candidates = list(row.get("candidates", []))
        candidate_ids = {str(candidate["item_id"]) for candidate in candidates}
        if not target_item_id or target_item_id not in candidate_ids:
            continue

        stats = candidate_pool_stats(candidates)
        user_payload = {
            "user_id": user_id,
            "target_item_id": target_item_id,
            "user_profile": reader.get_user_profile(user_id),
            "user_summary": reader.build_user_summary(user_id),
            "history_text": reader.render_history(user_id, max_events=20),
            "candidates": candidates,
            "candidate_stats": stats,
        }

        for scenario_id in eligible_scenarios:
            if scenario_is_suitable(scenario_id, stats, candidates, user_payload["user_summary"]):
                selected_by_scenario[scenario_id].append(dict(user_payload))

    scenario_counts = {scenario_id: len(rows) for scenario_id, rows in selected_by_scenario.items()}
    insufficient = {scenario_id: count for scenario_id, count in scenario_counts.items() if count < users_per_scenario}
    if insufficient:
        raise ValueError(
            f"Need {users_per_scenario} suitable ML-1M users per scenario, found {scenario_counts}."
        )

    requests: List[Dict[str, Any]] = []
    for scenario_id in eligible_scenarios:
        constraint_text = scenario_lookup[scenario_id]["constraint_text"]
        scenario_type = scenario_type_lookup[scenario_id]
        for index, selected in enumerate(selected_by_scenario[scenario_id][:users_per_scenario]):
            request_id = f"{scenario_id}-{index:03d}"
            requests.append(
                {
                    "request_id": request_id,
                    "dataset_name": "ml-1m",
                    "scenario_type": scenario_type,
                    "scenario_id": scenario_id,
                    "user_id": selected["user_id"],
                    "user_profile": selected["user_profile"],
                    "user_summary": selected["user_summary"],
                    "history_text": selected["history_text"],
                    "constraint_text": constraint_text,
                    "target_item_id": selected["target_item_id"],
                    "schema_fields": list(spec["schema_fields"]),
                    "reference_dsl": reference_builder(request_id, selected["user_summary"], scenario_id),
                    "candidates": selected["candidates"],
                    "candidate_stats": selected["candidate_stats"],
                }
            )
    return requests


def export_ml1m_suitable_requests(
    dataset_dir: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    users_per_scenario: int = 100,
    min_horror_candidates: int = 1,
    min_children_candidates: int = 2,
    min_comedy_candidates: int = 3,
    min_dominant_genres: int = 3,
) -> List[Dict[str, Any]]:
    reader = get_reader(dataset_dir)
    candidate_rows = load_jsonl(candidates_path)
    requests = build_ml1m_suitable_requests(
        candidate_rows,
        reader,
        users_per_scenario=users_per_scenario,
        min_horror_candidates=min_horror_candidates,
        min_children_candidates=min_children_candidates,
        min_comedy_candidates=min_comedy_candidates,
        min_dominant_genres=min_dominant_genres,
    )
    write_jsonl(output_path, requests)
    return requests


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset/ml-1m")
    parser.add_argument("--candidates", default="outputs/ml1m_candidates_sasrec.jsonl")
    parser.add_argument("--output", default="outputs/ml1m_suitable_requests_600.jsonl")
    parser.add_argument("--users-per-scenario", type=int, default=100)
    parser.add_argument("--min-horror-candidates", type=int, default=1)
    parser.add_argument("--min-children-candidates", type=int, default=2)
    parser.add_argument("--min-comedy-candidates", type=int, default=3)
    parser.add_argument("--min-dominant-genres", type=int, default=3)
    args = parser.parse_args()

    export_ml1m_suitable_requests(
        dataset_dir=args.dataset_dir,
        candidates_path=args.candidates,
        output_path=args.output,
        users_per_scenario=args.users_per_scenario,
        min_horror_candidates=args.min_horror_candidates,
        min_children_candidates=args.min_children_candidates,
        min_comedy_candidates=args.min_comedy_candidates,
        min_dominant_genres=args.min_dominant_genres,
    )


if __name__ == "__main__":
    main()
