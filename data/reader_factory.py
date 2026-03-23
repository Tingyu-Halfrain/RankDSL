from __future__ import annotations

from pathlib import Path

from .amazon_books_reader import AmazonBooksReader
from .ml1m_reader import ML1MReader


def get_dataset_name(dataset_dir: str | Path) -> str:
    return Path(dataset_dir).name


def get_reader(dataset_dir: str | Path, semantic_cache_path: str | Path | None = None):
    dataset_name = get_dataset_name(dataset_dir)
    if dataset_name == "ml-1m":
        return ML1MReader(dataset_dir)
    if dataset_name == "amazon-books":
        return AmazonBooksReader(dataset_dir, semantic_cache_path=semantic_cache_path)
    raise ValueError(f"Unsupported dataset: {dataset_name}")

