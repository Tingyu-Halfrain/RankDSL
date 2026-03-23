from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _split_category_tokens(value: str) -> List[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    parts = [html.unescape(part.strip().strip("'").strip('"')) for part in cleaned.split(",")]
    return [part for part in parts if part]


def _derive_dominant_category(categories: List[str]) -> str:
    filtered = [category for category in categories if category != "Books"]
    source = filtered or categories
    return sorted(source)[0] if source else "UNKNOWN"


class AmazonBooksReader:
    def __init__(
        self,
        dataset_path: str | Path = "RankDSL/dataset/amazon-books",
        semantic_cache_path: str | Path | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.semantic_cache_path = Path(semantic_cache_path) if semantic_cache_path else None
        self.item_rows = self._load_atomic_file("item")
        self.inter_rows = self._load_atomic_file("inter")
        self.semantic_cache = self._load_semantic_cache()

        self.item_map, self.item_meta_map = self._build_item_maps()
        self.history_map = self._build_history_map()
        self.filtered_history_map = self._build_filtered_history_map()
        self.test_target_map = self._build_test_target_map()

    def _load_atomic_file(self, suffix: str) -> List[Dict[str, str]]:
        file_path = self.dataset_path / f"{self.dataset_name}.{suffix}"
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return list(reader)

    def _load_semantic_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.semantic_cache_path or not self.semantic_cache_path.exists():
            return {}
        with self.semantic_cache_path.open("r", encoding="utf-8") as handle:
            return {row["item_id"]: row for row in (json.loads(line) for line in handle if line.strip())}

    def _build_item_maps(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        item_text_map: Dict[str, str] = {}
        item_meta_map: Dict[str, Dict[str, Any]] = {}

        for row in self.item_rows:
            item_id = row["item_id:token"]
            title = html.unescape(row.get("title:token", "")).strip() or "Unknown Title"
            categories = _split_category_tokens(row.get("categories:token_seq", ""))
            brand = html.unescape(row.get("brand:token", "")).strip()
            price_raw = row.get("price:float", "").strip()
            price = float(price_raw) if price_raw else None
            dominant_category = _derive_dominant_category(categories)

            semantic = self.semantic_cache.get(item_id, {})
            description = html.unescape(semantic.get("description", "")).strip()
            authors = ", ".join(semantic.get("authors", [])[:3]) if semantic.get("authors") else ""
            category_text = ", ".join(categories[:6]) if categories else "Unknown"

            parts = [title]
            if authors:
                parts.append(f"by {authors}")
            parts.append(f"[{category_text}]")
            if brand:
                parts.append(f"brand: {brand}")
            if price is not None:
                parts.append(f"price: ${price:.2f}")
            if description:
                parts.append(f"description: {description[:300]}")

            item_text_map[item_id] = " | ".join(parts)
            item_meta_map[item_id] = {
                "item_id": item_id,
                "title": title,
                "genre": categories,
                "categories": categories,
                "dominant_genre": dominant_category,
                "dominant_category": dominant_category,
                "brand": brand or None,
                "price": price,
                "release_year": None,
                "semantic_description": description or None,
            }
        return item_text_map, item_meta_map

    def _build_history_map(self) -> Dict[str, List[Tuple[str, float, float]]]:
        per_user: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
        for row in self.inter_rows:
            per_user[row["user_id:token"]].append(
                (
                    row["item_id:token"],
                    float(row["rating:float"]),
                    float(row["timestamp:float"]),
                )
            )
        return {user_id: sorted(rows, key=lambda item: item[2]) for user_id, rows in per_user.items()}

    def _build_filtered_history_map(
        self,
        rating_threshold: float = 3.0,
        min_user_inter: int = 5,
        min_item_inter: int = 5,
    ) -> Dict[str, List[str]]:
        filtered_rows = [
            row for row in self.inter_rows if float(row["rating:float"]) >= rating_threshold
        ]

        changed = True
        while changed:
            user_counts = Counter(row["user_id:token"] for row in filtered_rows)
            item_counts = Counter(row["item_id:token"] for row in filtered_rows)
            updated = [
                row
                for row in filtered_rows
                if user_counts[row["user_id:token"]] >= min_user_inter
                and item_counts[row["item_id:token"]] >= min_item_inter
            ]
            changed = len(updated) != len(filtered_rows)
            filtered_rows = updated

        per_user: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for row in filtered_rows:
            per_user[row["user_id:token"]].append((row["item_id:token"], float(row["timestamp:float"])))
        return {user_id: [item_id for item_id, _ in sorted(rows, key=lambda item: item[1])] for user_id, rows in per_user.items()}

    def _build_test_target_map(self) -> Dict[str, str]:
        return {
            user_id: rows[-1][0]
            for user_id, rows in self.history_map.items()
            if len(rows) >= 3
        }

    def get_user_profile(self, user_id: str | int) -> str:
        return f"Amazon Books user {user_id} with historical purchase and rating behavior."

    def get_item_text(self, item_id: str | int) -> str:
        return self.item_map.get(str(item_id), "Unknown Book")

    def get_item_metadata(self, item_id: str | int) -> Dict[str, Any]:
        meta = self.item_meta_map.get(str(item_id))
        if meta is None:
            return {
                "item_id": str(item_id),
                "title": "Unknown Book",
                "genre": [],
                "categories": [],
                "dominant_genre": "UNKNOWN",
                "dominant_category": "UNKNOWN",
                "brand": None,
                "price": None,
                "release_year": None,
                "semantic_description": None,
            }
        return dict(meta)

    def get_item_genres(self, item_id: str | int) -> List[str]:
        return list(self.get_item_metadata(item_id)["categories"])

    def get_user_history(self, user_id: str | int) -> List[Tuple[str, float, float]]:
        return list(self.history_map.get(str(user_id), []))

    def render_history(self, user_id: str | int, max_events: int = 20) -> str:
        history = self.get_user_history(user_id)
        if not history:
            return "No history available."
        lines = []
        for item_id, rating, _ in history[-max_events:]:
            lines.append(f"- {self.get_item_text(item_id)} | rating: {rating:.1f}")
        return "\n".join(lines)

    def build_user_summary(self, user_id: str | int, top_categories: int = 2) -> str:
        category_counter = Counter()
        price_values: List[float] = []
        for item_id in self.filtered_history_map.get(str(user_id), []):
            metadata = self.get_item_metadata(item_id)
            category_counter.update(metadata["categories"])
            if metadata["price"] is not None:
                price_values.append(float(metadata["price"]))

        top_preferences = [category for category, _ in category_counter.most_common(top_categories) if category != "Books"]
        if not top_preferences:
            preference_text = "No strong historical category preference."
        elif len(top_preferences) == 1:
            preference_text = f"Historically prefers {top_preferences[0]} books."
        else:
            preference_text = f"Historically prefers {top_preferences[0]} and {top_preferences[1]} books."

        if not price_values:
            price_text = "Observed price preference is unknown."
        else:
            average_price = sum(price_values) / len(price_values)
            if average_price <= 8:
                price_text = "Often engages with lower-priced books."
            elif average_price <= 18:
                price_text = "Often engages with moderately priced books."
            else:
                price_text = "Often engages with higher-priced books."

        return f"{self.get_user_profile(user_id)} {preference_text} {price_text}"

    def render_candidates(self, candidate_ids: Iterable[str | int]) -> Tuple[str, Dict[int, str]]:
        lines = []
        tmpid_to_item: Dict[int, str] = {}
        for index, item_id in enumerate(candidate_ids, start=1):
            item_id_str = str(item_id)
            tmpid_to_item[index] = item_id_str
            lines.append(f"[{index}] {self.get_item_text(item_id_str)}")
        return "\n".join(lines), tmpid_to_item

    def eligible_users(self) -> List[str]:
        return sorted(set(self.filtered_history_map).intersection(self.test_target_map))

