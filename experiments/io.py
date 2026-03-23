from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_candidate_lookup(path: str | Path) -> Dict[str, Dict[str, Any]]:
    return {row["user_id"]: row for row in load_jsonl(path)}

