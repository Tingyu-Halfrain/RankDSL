# RankDSL

RankDSL is a constrained reranking benchmark and runtime for LLM-generated ranking programs.

## ML-1M setup

The ML-1M pipeline now uses 6 evaluation scenarios:

- `filter_horror`
- `quota_comedy`
- `quota_children`
- `quota_comedy_filter_horror`
- `diversity_dominant_genre`
- `quota_children_diversity_filter_horror`

`outputs/ml1m_suitable_requests_600.jsonl` means `6 scenarios x 100 feasible requests`, not 600 unique users.

The suitable-request generator is `data/select_suitable_users.py`. It filters SASRec top-20 candidate pools with scenario-specific thresholds and then runs `verify_dsl(...)` to ensure each request is feasible before export.

## Main pipeline

1. Train SASRec on ML-1M.
2. Export top-20 candidates with `export_ml1m_sasrec_candidates.py`.
3. Reuse `outputs/ml1m_suitable_requests_600.jsonl` through `--requests-file`.
4. Run experiments with `run_rankdsl_experiment.py`.

See [operation.md](/mnt/data/binbin/mmm/RankDSL/operation.md) for the exact commands.

## Result caching

`run_rankdsl_experiment.py` supports:

- `--save-results`: persist each request to `results/saved_rankings/{request_id}.json`
- `--load-from-cache`: load cached per-request outputs instead of calling the LLM again
- `--num-paraphrases`: configurable paraphrase count, default `3`

If DSL compilation fails, the runtime now falls back to `base_recall` instead of returning a hard-zero empty ranking.

## V2 metrics

Detailed offline metrics live in `evaluation/detailed_metrics.py`, including:

- `filter_ok`
- `quota_status`
- `sliding_window_ok`
- `max_rep_in_any_window`
- `sliding_violation_rate`
- `ild_score`

`evaluation/metrics.py` summary now also includes:

- `filter_satisfaction`
- `quota_satisfaction`
- `diversity_satisfaction`
- `sliding_window_ok_rate`
- `ild_score_avg`
- `ndcg_only_successful`

## Offline analysis

- Batch recompute metrics: `python analysis/compute_metrics_offline.py --input-dir results/saved_rankings --output summary_detailed.json`
- Notebook entrypoint: `analysis/inspect_results.ipynb`
