from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RankDSL.core.runtime import load_ranking_from_disk, save_ranking_to_disk
from RankDSL.evaluation.detailed_metrics import enrich_request_result
from RankDSL.evaluation.metrics import calculate_summary


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/saved_rankings")
    parser.add_argument("--output", default="summary_detailed.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    results = [enrich_request_result(load_ranking_from_disk(path)) for path in files]
    payload = {
        "input_dir": str(input_dir),
        "files_processed": len(files),
        "summary": calculate_summary(results),
    }
    save_ranking_to_disk(args.output, payload)


if __name__ == "__main__":
    main()
