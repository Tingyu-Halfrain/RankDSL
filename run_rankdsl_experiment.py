from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RankDSL.experiments.runner import run_experiment


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset/ml-1m")
    parser.add_argument("--requests", default="outputs/ml1m_requests.jsonl")
    parser.add_argument("--candidates", default="outputs/ml1m_candidates_popularity.jsonl")
    parser.add_argument("--output", default="outputs/experiment_results.json")
    parser.add_argument("--candidate-topn", type=int, default=20)
    parser.add_argument("--llm-mode", default="stub", choices=["stub", "api"])
    parser.add_argument("--semantic-cache", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--scenario-size", type=int, default=50)
    parser.add_argument("--max-eval-users", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--allow-miss-users", action="store_true")
    parser.add_argument("--llm-log-path", default=None)
    parser.add_argument("--llm-parse-log-path", default="outputs/llm_parse_debug.jsonl")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_experiment(
        dataset_dir=args.dataset_dir,
        requests_path=args.requests,
        candidates_path=args.candidates,
        output_path=args.output,
        scenario_size=args.scenario_size,
        candidate_topn=args.candidate_topn,
        llm_mode=args.llm_mode,
        semantic_cache_path=args.semantic_cache,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_eval_users=args.max_eval_users,
        hit_users_only=not args.allow_miss_users,
        sample_seed=args.sample_seed,
        show_progress=not args.quiet,
        llm_log_path=args.llm_log_path,
        llm_parse_log_path=args.llm_parse_log_path,
    )


if __name__ == "__main__":
    main()
