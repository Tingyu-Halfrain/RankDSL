from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RankDSL.data.recbole_export import export_sasrec_topk_candidates
from RankDSL.experiments.candidate_builder import ensure_candidate_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="RankDSL/configs/sasrec_amazonbooks.yaml")
    parser.add_argument("--checkpoint", default="RankDSL/saved_ckpt/SASRec_amazonbooks_top50.pth")
    parser.add_argument("--dataset-dir", default="RankDSL/dataset/amazon-books")
    parser.add_argument("--semantic-cache", default="RankDSL/outputs/amazon_books_semantics.jsonl")
    parser.add_argument("--output", default="RankDSL/outputs/amazon_books_candidates_sasrec.jsonl")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    try:
        export_sasrec_topk_candidates(args.config, args.checkpoint, args.output, k=args.topk)
        ensure_candidate_metadata(args.dataset_dir, args.output, semantic_cache_path=args.semantic_cache)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()

