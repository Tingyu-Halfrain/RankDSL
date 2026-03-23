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
    parser.add_argument("--config", default="RankDSL/configs/sasrec_ml1m.yaml")
    parser.add_argument("--checkpoint", default="RankDSL/saved_ckpt/SASRec_ml1m_top50.pth")
    parser.add_argument("--dataset-dir", default="RankDSL/dataset/ml-1m")
    parser.add_argument("--output", default="RankDSL/outputs/ml1m_candidates_sasrec.jsonl")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    try:
        export_sasrec_topk_candidates(args.config, args.checkpoint, args.output, k=args.topk, dataset_name="ml-1m")
        ensure_candidate_metadata(args.dataset_dir, args.output)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
