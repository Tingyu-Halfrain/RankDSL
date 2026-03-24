from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..data.reader_factory import get_dataset_name, get_reader
from .dataset_specs import get_dataset_spec
from .io import write_jsonl


def select_request_users(
    eligible_users: Sequence[str],
    reader: Any,
    candidate_lookup: Dict[str, Dict[str, Any]] | None = None,
    hit_users_only: bool = False,
) -> List[str]:
    if not hit_users_only or candidate_lookup is None:
        return list(eligible_users)

    selected_users: List[str] = []
    for user_id in eligible_users:
        candidate_row = candidate_lookup.get(user_id)
        if not candidate_row:
            continue
        target_item_id = reader.test_target_map.get(user_id)
        if target_item_id is None:
            continue
        candidate_ids = {candidate["item_id"] for candidate in candidate_row["candidates"]}
        if target_item_id in candidate_ids:
            selected_users.append(user_id)
    return selected_users


def generate_requests(
    dataset_dir: str | Path,
    scenario_size: int = 50,
    seed: int = 2026,
    semantic_cache_path: str | Path | None = None,
    candidate_lookup: Dict[str, Dict[str, Any]] | None = None,
    hit_users_only: bool = False,
) -> List[Dict[str, Any]]:
    dataset_name = get_dataset_name(dataset_dir)
    spec = get_dataset_spec(dataset_name)
    reader = get_reader(dataset_dir, semantic_cache_path=semantic_cache_path)
    eligible_users = reader.eligible_users()
    random.Random(seed).shuffle(eligible_users)
    eligible_users = select_request_users(
        eligible_users,
        reader,
        candidate_lookup=candidate_lookup,
        hit_users_only=hit_users_only,
    )

    needed = scenario_size * len(spec["scenarios"])
    if len(eligible_users) < needed:
        raise ValueError(f"Need at least {needed} eligible users, got {len(eligible_users)}")

    requests: List[Dict[str, Any]] = []
    cursor = 0
    for scenario in spec["scenarios"]:
        for offset, user_id in enumerate(eligible_users[cursor : cursor + scenario_size]):
            request_id = f"{scenario['scenario_id']}-{offset:03d}"
            user_summary = reader.build_user_summary(user_id)
            requests.append(
                {
                    "request_id": request_id,
                    "dataset_name": dataset_name,
                    "scenario_id": scenario["scenario_id"],
                    "user_id": user_id,
                    "user_profile": reader.get_user_profile(user_id),
                    "user_summary": user_summary,
                    "history_text": reader.render_history(user_id, max_events=20),
                    "constraint_text": scenario["constraint_text"],
                    "target_item_id": reader.test_target_map[user_id],
                    "schema_fields": spec["schema_fields"],
                    "reference_dsl": spec["reference_builder"](request_id, user_summary, scenario["scenario_id"]),
                }
            )
        cursor += scenario_size
    return requests


def export_requests(
    dataset_dir: str | Path,
    output_path: str | Path,
    scenario_size: int = 50,
    seed: int = 2026,
    semantic_cache_path: str | Path | None = None,
    candidate_lookup: Dict[str, Dict[str, Any]] | None = None,
    hit_users_only: bool = False,
) -> None:
    write_jsonl(
        output_path,
        generate_requests(
            dataset_dir,
            scenario_size=scenario_size,
            seed=seed,
            semantic_cache_path=semantic_cache_path,
            candidate_lookup=candidate_lookup,
            hit_users_only=hit_users_only,
        ),
    )
