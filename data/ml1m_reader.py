from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.runtime import derive_dominant_genre


GENDER_MAP = {
    "M": "Male",
    "F": "Female",
}

AGE_MAP = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "over 56",
}

OCCUPATION_MAP = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


class ML1MReader:
    def __init__(self, dataset_path: str | Path = "RankDSL/dataset/ml-1m"):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.item_rows = self._load_atomic_file("item")
        self.user_rows = self._load_atomic_file("user")
        self.inter_rows = self._load_atomic_file("inter")

        self.item_map, self.item_genre_map, self.item_meta_map = self._build_item_maps()
        self.user_map = self._build_user_map()
        self.history_map = self._build_history_map()
        self.filtered_history_map = self._build_filtered_history_map()
        self.test_target_map = self._build_test_target_map()

    def _load_atomic_file(self, suffix: str) -> List[Dict[str, str]]:
        file_path = self.dataset_path / f"{self.dataset_name}.{suffix}"
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return list(reader)

    @staticmethod
    def _normalize_genres(genre_value: Any) -> List[str]:
        if genre_value is None:
            return []
        text = str(genre_value).strip()
        if not text or text.lower() == "nan":
            return []
        return [part for part in text.replace("|", " ").split() if part]

    def _build_item_maps(self) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
        item_text_map: Dict[str, str] = {}
        item_genre_map: Dict[str, List[str]] = {}
        item_meta_map: Dict[str, Dict[str, Any]] = {}

        for row in self.item_rows:
            item_id = row["item_id:token"]
            title = row["movie_title:token_seq"]
            year = int(row["release_year:token"])
            genres = self._normalize_genres(row["genre:token_seq"])
            genre_text = ", ".join(genres) if genres else "Unknown"

            item_text_map[item_id] = f"{title} ({year}) [{genre_text}]"
            item_genre_map[item_id] = genres
            item_meta_map[item_id] = {
                "item_id": item_id,
                "title": title,
                "genre": genres,
                "dominant_genre": derive_dominant_genre(genres),
                "release_year": year,
            }
        return item_text_map, item_genre_map, item_meta_map

    def _build_user_map(self) -> Dict[str, str]:
        user_map: Dict[str, str] = {}
        for row in self.user_rows:
            user_id = row["user_id:token"]
            gender = GENDER_MAP.get(row["gender:token"], "Unknown")
            age = AGE_MAP.get(int(row["age:token"]), "Unknown")
            occupation = OCCUPATION_MAP.get(int(row["occupation:token"]), "other")
            user_map[user_id] = f"The user is a {gender} who is {age} years old and works as a {occupation}."
        return user_map

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

        history_map: Dict[str, List[Tuple[str, float, float]]] = {}
        for user_id, rows in per_user.items():
            history_map[user_id] = sorted(rows, key=lambda item: item[2])
        return history_map

    def _build_filtered_history_map(
        self,
        rating_threshold: float = 4.0,
        min_user_inter: int = 10,
        min_item_inter: int = 10,
    ) -> Dict[str, List[str]]:
        filtered_rows = [
            row
            for row in self.inter_rows
            if float(row["rating:float"]) >= rating_threshold
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

        filtered_history_map: Dict[str, List[str]] = {}
        for user_id, rows in per_user.items():
            filtered_history_map[user_id] = [item_id for item_id, _ in sorted(rows, key=lambda item: item[1])]
        return filtered_history_map

    def _build_test_target_map(self) -> Dict[str, str]:
        return {
            user_id: rows[-1][0]
            for user_id, rows in self.history_map.items()
            if len(rows) >= 3
        }

    def get_user_profile(self, user_id: str | int) -> str:
        return self.user_map.get(str(user_id), "Unknown User")

    def get_item_text(self, item_id: str | int) -> str:
        return self.item_map.get(str(item_id), "Unknown Item")

    def get_item_genres(self, item_id: str | int) -> List[str]:
        return list(self.item_genre_map.get(str(item_id), []))

    def get_item_metadata(self, item_id: str | int) -> Dict[str, Any]:
        meta = self.item_meta_map.get(str(item_id))
        if meta is None:
            return {
                "item_id": str(item_id),
                "title": "Unknown Item",
                "genre": [],
                "dominant_genre": "UNKNOWN",
                "release_year": None,
            }
        return dict(meta)

    def get_user_history(self, user_id: str | int) -> List[Tuple[str, float, float]]:
        return list(self.history_map.get(str(user_id), []))

    def get_filtered_user_history(self, user_id: str | int) -> List[str]:
        return list(self.filtered_history_map.get(str(user_id), []))

    def render_history(self, user_id: str | int, max_events: int = 20) -> str:
        history = self.get_user_history(user_id)
        if not history:
            return "No history available."
        lines = []
        for item_id, rating, _ in history[-max_events:]:
            lines.append(f"- {self.get_item_text(item_id)} | rating: {rating:.1f}")
        return "\n".join(lines)

    def render_candidates(self, candidate_ids: Iterable[str | int]) -> Tuple[str, Dict[int, str]]:
        lines: List[str] = []
        tmpid_to_item: Dict[int, str] = {}
        for index, item_id in enumerate(candidate_ids, start=1):
            item_id_str = str(item_id)
            tmpid_to_item[index] = item_id_str
            lines.append(f"[{index}] {self.get_item_text(item_id_str)}")
        return "\n".join(lines), tmpid_to_item

    def build_user_summary(self, user_id: str | int, top_genres: int = 2) -> str:
        genre_counter = Counter()
        for item_id in self.get_filtered_user_history(user_id):
            genre_counter.update(self.get_item_genres(item_id))
        favorite_genres = [genre for genre, _ in genre_counter.most_common(top_genres)]
        if not favorite_genres:
            preference_text = "No strong historical genre preference."
        elif len(favorite_genres) == 1:
            preference_text = f"Historically prefers {favorite_genres[0]} titles."
        else:
            preference_text = f"Historically prefers {favorite_genres[0]} and {favorite_genres[1]} titles."
        return f"{self.get_user_profile(user_id)} {preference_text}"

    def eligible_users(self) -> List[str]:
        return sorted(set(self.filtered_history_map).intersection(self.test_target_map))

