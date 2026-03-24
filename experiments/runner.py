from __future__ import annotations

import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from ..core.dsl_parser import canonicalize_dsl, parse_ranking_dsl
from ..core.runtime import build_group_memberships, normalize_candidates, ranking_genre_coverage, ranking_violations
from ..core.solver import GreedySolver, ILPSolver
from ..core.verifier import verify_dsl
from ..evaluation.metrics import aggregate_method, hit_at_10, ndcg_at_10
from ..llm.client import RankDSLLLMClient
from .baselines import align_candidate_order, parse_direct_rerank_ids, score_adjust_baseline
from .candidate_builder import build_popularity_candidates, ensure_candidate_metadata
from .dataset_specs import get_dataset_spec
from .io import load_candidate_lookup, load_jsonl
from .request_builder import export_requests
from ..data.reader_factory import get_dataset_name


def log_progress(message: str) -> None:
    print(f"[RankDSL] {message}", flush=True)


def top10_item_dicts(ranking) -> List[Dict[str, Any]]:
    return [
        {
            "item_id": candidate.item_id,
            "title": candidate.title,
            "genre": list(candidate.genre),
            "dominant_genre": candidate.dominant_genre,
            "release_year": candidate.release_year,
            "base_score": candidate.base_score,
        }
        for candidate in ranking[:10]
    ]


def needs_request_regeneration(
    dataset_dir: str | Path,
    requests_path: str | Path,
    scenario_size: int,
    candidate_lookup: Dict[str, Dict[str, Any]] | None = None,
    hit_users_only: bool = True,
) -> tuple[bool, str]:
    requests_path = Path(requests_path)
    if not requests_path.exists():
        return True, "missing"

    dataset_name = get_dataset_name(dataset_dir)
    spec = get_dataset_spec(dataset_name)
    expected_counts = Counter({scenario["scenario_id"]: scenario_size for scenario in spec["scenarios"]})
    expected_total = sum(expected_counts.values())

    try:
        requests = load_jsonl(requests_path)
    except Exception as exc:
        return True, f"unreadable ({exc})"

    if len(requests) != expected_total:
        return True, f"expected {expected_total} requests for scenario_size={scenario_size}, found {len(requests)}"

    scenario_counts = Counter(request.get("scenario_id") for request in requests)
    if scenario_counts != expected_counts:
        return True, (
            f"scenario distribution mismatch: expected {dict(expected_counts)}, "
            f"found {dict(scenario_counts)}"
        )

    if any(request.get("dataset_name") != dataset_name for request in requests):
        return True, f"dataset mismatch: expected dataset_name={dataset_name}"

    if hit_users_only and candidate_lookup is not None:
        for request in requests:
            candidate_row = candidate_lookup.get(request["user_id"])
            if not candidate_row:
                return True, f"request file contains users without candidates: user_id={request['user_id']}"
            candidate_ids = {candidate["item_id"] for candidate in candidate_row["candidates"]}
            if request["target_item_id"] not in candidate_ids:
                return True, (
                    "request file contains non-hit users while hit_users_only=True: "
                    f"request_id={request['request_id']}"
                )

    return False, "matches requested scenario_size"


def select_requests_for_evaluation(
    requests: List[Dict[str, Any]],
    candidate_lookup: Dict[str, Dict[str, Any]],
    max_users: int = 0,
    hit_users_only: bool = True,
    sample_seed: int = 2026,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eligible_requests: List[Dict[str, Any]] = []
    missing_candidate_users = 0
    miss_users = 0

    for request in requests:
        candidate_row = candidate_lookup.get(request["user_id"])
        if not candidate_row:
            missing_candidate_users += 1
            continue

        if hit_users_only:
            candidate_ids = {candidate["item_id"] for candidate in candidate_row["candidates"]}
            if request["target_item_id"] not in candidate_ids:
                miss_users += 1
                continue

        eligible_requests.append(request)

    rng = random.Random(sample_seed)
    if max_users > 0 and len(eligible_requests) > max_users:
        eligible_requests = rng.sample(eligible_requests, max_users)

    eligible_requests.sort(key=lambda row: row["request_id"])
    selection_info = {
        "requested_total": len(requests),
        "eligible_total": len(eligible_requests),
        "missing_candidate_users": missing_candidate_users,
        "filtered_miss_users": miss_users,
        "max_users": max_users,
        "hit_users_only": hit_users_only,
        "sample_seed": sample_seed,
    }
    return eligible_requests, selection_info


def run_request(
    request: Dict[str, Any],
    candidate_row: Dict[str, Any],
    client: RankDSLLLMClient,
    verbose: bool = False,
) -> Dict[str, Any]:
    candidates = candidate_row["candidates"]
    greedy_solver = GreedySolver()
    ilp_solver = ILPSolver()
    target_item_id = request["target_item_id"]

    result: Dict[str, Any] = {
        "request_id": request["request_id"],
        "user_id": request["user_id"],
        "scenario_id": request["scenario_id"],
        "target_item_id": target_item_id,
    }
    reference_dsl = parse_ranking_dsl(request.get("reference_dsl") or RankDSLLLMClient._stub_compile(request))
    reference_memberships = build_group_memberships(reference_dsl, normalize_candidates(candidates))

    base_ranking = normalize_candidates(candidates[:10])
    base_violations = ranking_violations(base_ranking, reference_dsl, reference_memberships)
    result["base_recall"] = {
        "hit@10": hit_at_10(base_ranking, target_item_id),
        "ndcg@10": ndcg_at_10(base_ranking, target_item_id),
        "constraint_satisfaction": float(not base_violations),
        "violation_count": len(base_violations),
        "genre_coverage": ranking_genre_coverage(base_ranking),
    }

    score_adjusted = normalize_candidates(score_adjust_baseline(request, candidates))
    score_adjust_violations = ranking_violations(score_adjusted, reference_dsl, reference_memberships)
    result["score_adjust_greedy"] = {
        "hit@10": hit_at_10(score_adjusted, target_item_id),
        "ndcg@10": ndcg_at_10(score_adjusted, target_item_id),
        "constraint_satisfaction": float(not score_adjust_violations),
        "violation_count": len(score_adjust_violations),
        "genre_coverage": ranking_genre_coverage(score_adjusted),
    }

    rankdsl_runs = []
    direct_runs = []
    canonical_programs = []
    compile_successes = 0

    for paraphrase_index in range(3):
        start = time.perf_counter()
        compile_response = client.compile_rankdsl(request, paraphrase_index=paraphrase_index)
        initial_raw_response = compile_response.text
        compile_payload: Any = compile_response.text
        initial_parse_error: str | None = None
        try:
            compile_payload = client.parse_json_response(
                compile_response.text,
                meta={
                    "interaction_type": "compile_rankdsl",
                    "request_id": request["request_id"],
                    "user_id": request["user_id"],
                    "scenario_id": request["scenario_id"],
                    "paraphrase_index": paraphrase_index,
                    "stage": "initial_compile",
                },
            )
        except Exception as exc:
            initial_parse_error = str(exc)
        verification = verify_dsl(compile_payload, candidates)
        client.log_debug_event(
            {
                "event": "compile_verification",
                "interaction_type": "compile_rankdsl",
                "request_id": request["request_id"],
                "user_id": request["user_id"],
                "scenario_id": request["scenario_id"],
                "paraphrase_index": paraphrase_index,
                "stage": "initial_compile",
                "verification_ok": verification.ok,
                "verification_errors": verification.errors,
            }
        )
        repair_raw_response: str | None = None
        repair_parse_error: str | None = None
        if not verification.ok:
            repair_error_payload = verification.errors[0] if verification.errors else {
                "code": "unknown_compile_error",
                "error_type": "schema",
                "message": "Compilation failed verification",
                "suggestion": "Return one corrected RankDSL JSON object only.",
            }
            compile_response = client.compile_rankdsl(
                request,
                paraphrase_index=paraphrase_index,
                repair_error=json.dumps(repair_error_payload, ensure_ascii=False, default=str),
            )
            repair_raw_response = compile_response.text
            compile_payload = compile_response.text
            try:
                compile_payload = client.parse_json_response(
                    compile_response.text,
                    meta={
                        "interaction_type": "compile_rankdsl",
                        "request_id": request["request_id"],
                        "user_id": request["user_id"],
                        "scenario_id": request["scenario_id"],
                        "paraphrase_index": paraphrase_index,
                        "stage": "repair_compile",
                    },
                )
            except Exception as exc:
                repair_parse_error = str(exc)
            verification = verify_dsl(compile_payload, candidates)
            client.log_debug_event(
                {
                    "event": "compile_verification",
                    "interaction_type": "compile_rankdsl",
                    "request_id": request["request_id"],
                    "user_id": request["user_id"],
                    "scenario_id": request["scenario_id"],
                    "paraphrase_index": paraphrase_index,
                    "stage": "repair_compile",
                    "verification_ok": verification.ok,
                    "verification_errors": verification.errors,
                }
            )
        latency = time.perf_counter() - start
        if verbose:
            status = "ok" if verification.ok else "compile_failed"
            log_progress(
                f"request={request['request_id']} paraphrase={paraphrase_index + 1}/3 compile_status={status} compile_latency={latency:.2f}s"
            )

        if verification.ok and verification.dsl is not None:
            compile_successes += 1
            canonical_programs.append(canonicalize_dsl(verification.dsl))
            greedy_result = greedy_solver.solve(verification.dsl, candidates)
            ilp_result = ilp_solver.solve(verification.dsl, candidates)
            greedy_reference_violations = ranking_violations(greedy_result.ranking, reference_dsl, reference_memberships)
            ilp_reference_violations = ranking_violations(ilp_result.ranking, reference_dsl, reference_memberships)
            rankdsl_runs.append(
                {
                    "latency": latency,
                    "greedy": {
                        "hit@10": hit_at_10(greedy_result.ranking, target_item_id),
                        "ndcg@10": ndcg_at_10(greedy_result.ranking, target_item_id),
                        "constraint_satisfaction": float(not greedy_reference_violations),
                        "violation_count": len(greedy_reference_violations),
                        "genre_coverage": ranking_genre_coverage(greedy_result.ranking),
                        "ranking": top10_item_dicts(greedy_result.ranking),
                        "solver": greedy_result.metadata.get("solver_effective", greedy_result.metadata.get("solver")),
                    },
                    "ilp": {
                        "hit@10": hit_at_10(ilp_result.ranking, target_item_id),
                        "ndcg@10": ndcg_at_10(ilp_result.ranking, target_item_id),
                        "constraint_satisfaction": float(not ilp_reference_violations),
                        "violation_count": len(ilp_reference_violations),
                        "genre_coverage": ranking_genre_coverage(ilp_result.ranking),
                        "ranking": top10_item_dicts(ilp_result.ranking),
                        "solver": ilp_result.metadata.get("solver_effective", ilp_result.metadata.get("solver")),
                        "solver_metadata": ilp_result.metadata,
                    },
                }
            )
        else:
            rankdsl_runs.append(
                {
                    "latency": latency,
                    "compile_error": verification.errors,
                    "compile_raw_response": initial_raw_response,
                    "compile_parse_error": initial_parse_error,
                    "repair_raw_response": repair_raw_response,
                    "repair_parse_error": repair_parse_error,
                    "greedy": {"constraint_satisfaction": 0.0, "violation_count": 1, "ranking": []},
                    "ilp": {"constraint_satisfaction": 0.0, "violation_count": 1, "ranking": []},
                }
            )

        direct_start = time.perf_counter()
        direct_response = client.direct_rerank(request, candidates, paraphrase_index=paraphrase_index)
        direct_latency = time.perf_counter() - direct_start
        try:
            ordered_ids = parse_direct_rerank_ids(client, direct_response.text)
            direct_ranking = align_candidate_order(ordered_ids, candidates)
            direct_violations = ranking_violations(direct_ranking, reference_dsl, reference_memberships)
            direct_runs.append(
                {
                    "latency": direct_latency,
                    "hit@10": hit_at_10(direct_ranking, target_item_id),
                    "ndcg@10": ndcg_at_10(direct_ranking, target_item_id),
                    "constraint_satisfaction": float(not direct_violations),
                    "violation_count": len(direct_violations),
                    "genre_coverage": ranking_genre_coverage(direct_ranking),
                    "ranking": top10_item_dicts(direct_ranking),
                }
            )
        except Exception:
            direct_runs.append(
                {
                    "latency": direct_latency,
                    "hit@10": 0.0,
                    "ndcg@10": 0.0,
                    "constraint_satisfaction": 0.0,
                    "violation_count": 1,
                    "genre_coverage": 0.0,
                    "ranking": [],
                }
            )

    result["rankdsl"] = {
        "compile_success_rate": compile_successes / 3.0,
        "canonical_program_agreement": float(len(set(canonical_programs)) == 1) if canonical_programs else 0.0,
        "constraint_satisfaction_variance": statistics.pvariance(
            [run["ilp"]["constraint_satisfaction"] for run in rankdsl_runs]
        )
        if rankdsl_runs
        else 0.0,
        "runs": rankdsl_runs,
    }
    result["prompt_only_direct_rerank"] = {
        "constraint_satisfaction_variance": statistics.pvariance(
            [run["constraint_satisfaction"] for run in direct_runs]
        )
        if direct_runs
        else 0.0,
        "runs": direct_runs,
    }
    if verbose:
        best_ilp = result["rankdsl"]["runs"][0]["ilp"] if result["rankdsl"]["runs"] else {}
        log_progress(
            f"request={request['request_id']} done compile_success_rate={result['rankdsl']['compile_success_rate']:.2f} "
            f"ilp_constraint={best_ilp.get('constraint_satisfaction', 0.0):.2f} "
            f"ilp_ndcg={best_ilp.get('ndcg@10', 0.0):.4f}"
        )
    return result


def run_experiment(
    dataset_dir: str | Path,
    requests_path: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    scenario_size: int = 50,
    candidate_topn: int = 20,
    llm_mode: str = "stub",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "claude-opus-4-6",
    semantic_cache_path: str | Path | None = None,
    max_eval_users: int = 0,
    hit_users_only: bool = True,
    sample_seed: int = 2026,
    show_progress: bool = True,
    llm_log_path: str | Path | None = None,
    llm_parse_log_path: str | Path | None = None,
) -> Dict[str, Any]:
    requests_path = Path(requests_path)
    candidates_path = Path(candidates_path)
    output_path = Path(output_path)

    if not candidates_path.exists():
        if show_progress:
            log_progress(f"candidates file missing, building candidates at {candidates_path}")
        build_popularity_candidates(
            dataset_dir,
            candidates_path,
            top_n=candidate_topn,
            semantic_cache_path=semantic_cache_path,
        )
    elif show_progress:
        log_progress(f"using existing candidates file {candidates_path}")

    if show_progress:
        log_progress("hydrating candidate metadata")
    ensure_candidate_metadata(dataset_dir, candidates_path, semantic_cache_path=semantic_cache_path)

    candidate_lookup = load_candidate_lookup(candidates_path)
    should_regenerate_requests, request_reason = needs_request_regeneration(
        dataset_dir,
        requests_path,
        scenario_size,
        candidate_lookup=candidate_lookup,
        hit_users_only=hit_users_only,
    )
    if should_regenerate_requests:
        if show_progress:
            log_progress(f"generating requests at {requests_path} ({request_reason})")
        export_requests(
            dataset_dir,
            requests_path,
            scenario_size=scenario_size,
            semantic_cache_path=semantic_cache_path,
            candidate_lookup=candidate_lookup,
            hit_users_only=hit_users_only,
        )
    elif show_progress:
        log_progress(f"using existing requests file {requests_path} ({request_reason})")

    if show_progress:
        log_progress("loading requests and candidates")
    requests = load_jsonl(requests_path)
    selected_requests, selection_info = select_requests_for_evaluation(
        requests,
        candidate_lookup,
        max_users=max_eval_users,
        hit_users_only=hit_users_only,
        sample_seed=sample_seed,
    )
    if show_progress:
        log_progress(
            "selection "
            f"requested={selection_info['requested_total']} "
            f"eligible={selection_info['eligible_total']} "
            f"missing_candidates={selection_info['missing_candidate_users']} "
            f"filtered_miss={selection_info['filtered_miss_users']} "
            f"max_eval_users={selection_info['max_users']} "
            f"hit_users_only={selection_info['hit_users_only']}"
        )
    client = RankDSLLLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        mode=llm_mode,
        log_path=llm_log_path,
        parse_log_path=llm_parse_log_path,
    )
    if show_progress and llm_log_path:
        log_progress(f"writing llm interaction log to {llm_log_path}")
    if show_progress and llm_parse_log_path:
        log_progress(f"writing llm parse debug log to {llm_parse_log_path}")

    results = []
    total_selected = len(selected_requests)
    for index, request in enumerate(selected_requests, start=1):
        candidate_row = candidate_lookup.get(request["user_id"])
        if candidate_row:
            if show_progress:
                log_progress(
                    f"evaluating {index}/{total_selected} request={request['request_id']} "
                    f"user={request['user_id']} scenario={request['scenario_id']}"
                )
            results.append(run_request(request, candidate_row, client, verbose=show_progress))

    if not results:
        summary = {
            "base_recall": {},
            "score_adjust_greedy": {},
            "prompt_only_direct_rerank": {},
            "rankdsl_greedy": {},
            "rankdsl_ilp": {},
            "compile_success_rate": 0.0,
            "canonical_program_agreement": 0.0,
            "rankdsl_constraint_satisfaction_variance": 0.0,
            "prompt_only_constraint_satisfaction_variance": 0.0,
        }
        payload = {"selection": selection_info, "summary": summary, "results": results}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        if show_progress:
            log_progress(f"no eligible requests were evaluated; wrote empty result to {output_path}")
        return payload

    summary = {
        "base_recall": aggregate_method(results, lambda row: row["base_recall"]),
        "score_adjust_greedy": aggregate_method(results, lambda row: row["score_adjust_greedy"]),
        "prompt_only_direct_rerank": aggregate_method(results, lambda row: row["prompt_only_direct_rerank"]["runs"]),
        "rankdsl_greedy": aggregate_method(results, lambda row: [run["greedy"] for run in row["rankdsl"]["runs"]]),
        "rankdsl_ilp": aggregate_method(results, lambda row: [run["ilp"] for run in row["rankdsl"]["runs"]]),
        "compile_success_rate": sum(row["rankdsl"]["compile_success_rate"] for row in results) / len(results),
        "canonical_program_agreement": sum(row["rankdsl"]["canonical_program_agreement"] for row in results) / len(results),
        "rankdsl_constraint_satisfaction_variance": sum(
            row["rankdsl"]["constraint_satisfaction_variance"] for row in results
        )
        / len(results),
        "prompt_only_constraint_satisfaction_variance": sum(
            row["prompt_only_direct_rerank"]["constraint_satisfaction_variance"] for row in results
        )
        / len(results),
    }

    payload = {"selection": selection_info, "summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    if show_progress:
        log_progress(
            f"finished evaluation users={len(results)} "
            f"compile_success_rate={summary['compile_success_rate']:.4f} "
            f"rankdsl_ilp_constraint={summary['rankdsl_ilp'].get('constraint_satisfaction', 0.0):.4f} "
            f"output={output_path}"
        )
    return payload
