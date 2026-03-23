from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_HEADERS = {
    "User-Agent": "RankDSLBookEnricher/1.0",
}


def _http_get_json(url: str, timeout: int = 20) -> Any:
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def fetch_openlibrary_metadata(identifier: str) -> Dict[str, Any]:
    identifier = identifier.strip()
    if not identifier:
        return {}

    bib_key = urllib.parse.quote(f"ISBN:{identifier}")
    url = f"https://openlibrary.org/api/books?bibkeys={bib_key}&format=json&jscmd=data"
    payload = _http_get_json(url)
    record = payload.get(f"ISBN:{identifier}", {})
    if not record:
        return {}
    description = record.get("notes")
    if isinstance(description, dict):
        description = description.get("value")
    return {
        "source": "openlibrary",
        "title": record.get("title"),
        "authors": [author.get("name") for author in record.get("authors", []) if author.get("name")],
        "description": description,
        "categories": [subject.get("name") for subject in record.get("subjects", []) if subject.get("name")][:10],
        "published_date": record.get("publish_date"),
    }


def fetch_google_books_metadata(identifier: str, title: str | None = None) -> Dict[str, Any]:
    query = f"isbn:{identifier}" if identifier else ""
    if not query and title:
        query = f"intitle:{title}"
    elif title:
        query = f"{query}+intitle:{title}"
    url = "https://www.googleapis.com/books/v1/volumes?q=" + urllib.parse.quote(query) + "&maxResults=1"
    payload = _http_get_json(url)
    items = payload.get("items") or []
    if not items:
        return {}
    info = items[0].get("volumeInfo", {})
    return {
        "source": "google_books",
        "title": info.get("title"),
        "authors": info.get("authors", []),
        "description": info.get("description"),
        "categories": info.get("categories", []),
        "published_date": info.get("publishedDate"),
    }


def merge_book_metadata(base_row: Dict[str, Any], fetched: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_row)
    for key in ("title", "description", "published_date"):
        if fetched.get(key) and not merged.get(key):
            merged[key] = fetched[key]
    for key in ("authors", "categories"):
        if fetched.get(key) and not merged.get(key):
            merged[key] = fetched[key]
    merged["semantic_source"] = fetched.get("source") or merged.get("semantic_source")
    return merged


def enrich_book_rows(
    rows: Iterable[Dict[str, Any]],
    sleep_seconds: float = 0.2,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        base = {
            "item_id": row["item_id"],
            "title": row.get("title"),
            "description": row.get("description"),
            "authors": row.get("authors", []),
            "categories": row.get("categories", []),
            "published_date": row.get("published_date"),
            "semantic_source": row.get("semantic_source"),
        }
        fetched = fetch_openlibrary_metadata(row["item_id"])
        if not fetched:
            fetched = fetch_google_books_metadata(row["item_id"], title=row.get("title"))
        enriched.append(merge_book_metadata(base, fetched))
        time.sleep(sleep_seconds)
    return enriched


def write_enriched_rows(output_path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

