from __future__ import annotations

from typing import Any, Dict, List


ML1M_ALLOWED_FIELDS = ["genre", "dominant_genre", "release_year"]
AMAZON_ALLOWED_FIELDS = ["categories", "dominant_category", "price", "brand"]


def ml1m_reference_program(request_id: str, user_summary: str, scenario_id: str) -> Dict[str, Any]:
    mapping = {
        "filter_horror": {
            "groups": [{"group_id": "horror", "filter_expression": "genre == 'Horror'"}],
            "constraints": {"filters": [{"action": "exclude", "target_group": "horror"}], "quotas": [], "diversity": []},
            "objective": {"base_score_weight": 1.0, "group_boosts": []},
        },
        "quota_comedy": {
            "groups": [{"group_id": "comedy", "filter_expression": "genre == 'Comedy'"}],
            "constraints": {"filters": [], "quotas": [{"target_group": "comedy", "min_count": 3}], "diversity": []},
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "comedy", "weight": 0.2}]},
        },
        "quota_children": {
            "groups": [{"group_id": "children", "filter_expression": 'genre == "Children\'s"'}],
            "constraints": {"filters": [], "quotas": [{"target_group": "children", "min_count": 2}], "diversity": []},
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "children", "weight": 0.2}]},
        },
        "quota_comedy_filter_horror": {
            "groups": [
                {"group_id": "comedy", "filter_expression": "genre == 'Comedy'"},
                {"group_id": "horror", "filter_expression": "genre == 'Horror'"},
            ],
            "constraints": {
                "filters": [{"action": "exclude", "target_group": "horror"}],
                "quotas": [{"target_group": "comedy", "min_count": 3}],
                "diversity": [],
            },
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "comedy", "weight": 0.2}]},
        },
        "diversity_dominant_genre": {
            "groups": [],
            "constraints": {
                "filters": [],
                "quotas": [],
                "diversity": [{"attribute": "dominant_genre", "window_size": 3, "max_repetition": 1}],
            },
            "objective": {"base_score_weight": 1.0, "group_boosts": []},
        },
        "quota_children_diversity_filter_horror": {
            "groups": [
                {"group_id": "children", "filter_expression": 'genre == "Children\'s"'},
                {"group_id": "horror", "filter_expression": "genre == 'Horror'"},
            ],
            "constraints": {
                "filters": [{"action": "exclude", "target_group": "horror"}],
                "quotas": [{"target_group": "children", "min_count": 2}],
                "diversity": [{"attribute": "dominant_genre", "window_size": 3, "max_repetition": 1}],
            },
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "children", "weight": 0.2}]},
        },
    }
    base = mapping[scenario_id]
    return {
        "meta": {"request_id": request_id, "user_summary": user_summary, "top_k": 10},
        "groups": base["groups"],
        "constraints": base["constraints"],
        "objective": base["objective"],
        "tie_break": ["base_score desc", "item_id asc"],
    }


def amazon_reference_program(request_id: str, user_summary: str, scenario_id: str) -> Dict[str, Any]:
    mapping = {
        "filter_expensive": {
            "groups": [{"group_id": "expensive", "filter_expression": "price > 20"}],
            "constraints": {"filters": [{"action": "exclude", "target_group": "expensive"}], "quotas": [], "diversity": []},
            "objective": {"base_score_weight": 1.0, "group_boosts": []},
        },
        "quota_mystery": {
            "groups": [{"group_id": "mystery", "filter_expression": "categories == 'Mystery'"}],
            "constraints": {"filters": [], "quotas": [{"target_group": "mystery", "min_count": 3}], "diversity": []},
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "mystery", "weight": 0.2}]},
        },
        "quota_scifi": {
            "groups": [{"group_id": "scifi", "filter_expression": "categories == 'Science Fiction'"}],
            "constraints": {"filters": [], "quotas": [{"target_group": "scifi", "min_count": 2}], "diversity": []},
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "scifi", "weight": 0.2}]},
        },
        "quota_mystery_filter_expensive": {
            "groups": [
                {"group_id": "mystery", "filter_expression": "categories == 'Mystery'"},
                {"group_id": "expensive", "filter_expression": "price > 20"},
            ],
            "constraints": {
                "filters": [{"action": "exclude", "target_group": "expensive"}],
                "quotas": [{"target_group": "mystery", "min_count": 3}],
                "diversity": [],
            },
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "mystery", "weight": 0.2}]},
        },
        "diversity_dominant_category": {
            "groups": [],
            "constraints": {
                "filters": [],
                "quotas": [],
                "diversity": [{"attribute": "dominant_category", "window_size": 3, "max_repetition": 1}],
            },
            "objective": {"base_score_weight": 1.0, "group_boosts": []},
        },
        "quota_scifi_diversity_filter_expensive": {
            "groups": [
                {"group_id": "scifi", "filter_expression": "categories == 'Science Fiction'"},
                {"group_id": "expensive", "filter_expression": "price > 20"},
            ],
            "constraints": {
                "filters": [{"action": "exclude", "target_group": "expensive"}],
                "quotas": [{"target_group": "scifi", "min_count": 2}],
                "diversity": [{"attribute": "dominant_category", "window_size": 3, "max_repetition": 1}],
            },
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "scifi", "weight": 0.2}]},
        },
    }
    base = mapping[scenario_id]
    return {
        "meta": {"request_id": request_id, "user_summary": user_summary, "top_k": 10},
        "groups": base["groups"],
        "constraints": base["constraints"],
        "objective": base["objective"],
        "tie_break": ["base_score desc", "item_id asc"],
    }


def get_dataset_spec(dataset_name: str) -> Dict[str, Any]:
    if dataset_name == "ml-1m":
        scenarios = [
            ("filter_horror", "Top-10 must exclude Horror titles."),
            ("quota_comedy", "Top-10 must contain at least 3 Comedy titles."),
            ("quota_children", "Top-10 must contain at least 2 Children's titles."),
            ("quota_comedy_filter_horror", "Top-10 must contain at least 3 Comedy titles and exclude Horror titles."),
            ("diversity_dominant_genre", "Enforce diversity: within any window of 3 ranked items, dominant_genre cannot repeat."),
            ("quota_children_diversity_filter_horror", "Top-10 must contain at least 2 Children's titles, exclude Horror titles, and within any window of 3 ranked items dominant_genre cannot repeat."),
        ]
        return {
            "dataset_name": dataset_name,
            "schema_fields": ML1M_ALLOWED_FIELDS,
            "scenarios": [
                {"scenario_id": sid, "constraint_text": text}
                for sid, text in scenarios
            ],
            "reference_builder": ml1m_reference_program,
        }
    if dataset_name == "amazon-books":
        scenarios = [
            ("filter_expensive", "Top-10 must exclude books priced above $20."),
            ("quota_mystery", "Top-10 must contain at least 3 Mystery books."),
            ("quota_scifi", "Top-10 must contain at least 2 Science Fiction books."),
            ("quota_mystery_filter_expensive", "Top-10 must contain at least 3 Mystery books and exclude books priced above $20."),
            ("diversity_dominant_category", "Enforce diversity: within any window of 3 ranked books, dominant_category cannot repeat."),
            ("quota_scifi_diversity_filter_expensive", "Top-10 must contain at least 2 Science Fiction books, exclude books priced above $20, and within any window of 3 ranked books dominant_category cannot repeat."),
        ]
        return {
            "dataset_name": dataset_name,
            "schema_fields": AMAZON_ALLOWED_FIELDS,
            "scenarios": [
                {"scenario_id": sid, "constraint_text": text}
                for sid, text in scenarios
            ],
            "reference_builder": amazon_reference_program,
        }
    raise ValueError(f"Unsupported dataset spec: {dataset_name}")

