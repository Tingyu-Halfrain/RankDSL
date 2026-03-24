from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RankDSL.data.recbole_export import export_sasrec_topk_candidates
from RankDSL.data.select_suitable_users import export_ml1m_suitable_requests
from RankDSL.experiments.candidate_builder import ensure_candidate_metadata


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sasrec_ml1m.yaml")
    parser.add_argument("--checkpoint", default="saved_ckpt/SASRec_ml1m_top50.pth")
    parser.add_argument("--dataset-dir", default="dataset/ml-1m")
    parser.add_argument("--output", default="outputs/ml1m_candidates_sasrec.jsonl")
    parser.add_argument("--suitable-requests-output", default="outputs/ml1m_suitable_requests_600.jsonl")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--users-per-scenario", type=int, default=100)
    args = parser.parse_args()

    try:
        export_sasrec_topk_candidates(args.config, args.checkpoint, args.output, k=args.topk, dataset_name="ml-1m")
        ensure_candidate_metadata(args.dataset_dir, args.output)
        export_ml1m_suitable_requests(
            dataset_dir=args.dataset_dir,
            candidates_path=args.output,
            output_path=args.suitable_requests_output,
            users_per_scenario=args.users_per_scenario,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
