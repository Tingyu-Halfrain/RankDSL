from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RankDSL.data.book_metadata_enricher import enrich_book_rows, write_enriched_rows


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--item-path", default="dataset/amazon-books/amazon-books.item")
    parser.add_argument("--output", default="outputs/amazon_books_semantics.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()

    item_path = Path(args.item_path)
    rows = []
    with item_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if index < args.offset:
                continue
            if len(rows) >= args.limit:
                break
            rows.append(
                {
                    "item_id": row["item_id:token"],
                    "title": html.unescape(row.get("title:token", "")).strip(),
                    "categories": [part.strip().strip("'") for part in row.get("categories:token_seq", "").split(",") if part.strip()],
                }
            )

    enriched = enrich_book_rows(rows, sleep_seconds=args.sleep_seconds)
    write_enriched_rows(args.output, enriched)


if __name__ == "__main__":
    main()
