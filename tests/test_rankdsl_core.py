import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from RankDSL.core.dsl_parser import canonicalize_dsl, parse_ranking_dsl
from RankDSL.core.solver import GreedySolver, ILPSolver
from RankDSL.core.verifier import verify_dsl
from RankDSL.experiments.io import write_jsonl
from RankDSL.experiments.request_builder import generate_requests
from RankDSL.experiments.runner import needs_request_regeneration, run_request, select_requests_for_evaluation
from RankDSL.llm.client import LLMResponse, RankDSLLLMClient


TOY_CANDIDATES = [
    {"item_id": "1", "title": "Funny A", "genre": ["Comedy"], "release_year": 2001, "base_score": 0.95},
    {"item_id": "2", "title": "Funny B", "genre": ["Comedy"], "release_year": 2002, "base_score": 0.91},
    {"item_id": "3", "title": "Kids A", "genre": ["Children's"], "release_year": 2000, "base_score": 0.89},
    {"item_id": "4", "title": "Drama A", "genre": ["Drama"], "release_year": 1999, "base_score": 0.87},
    {"item_id": "5", "title": "Horror A", "genre": ["Horror"], "release_year": 1998, "base_score": 0.86},
    {"item_id": "6", "title": "Kids B", "genre": ["Children's"], "release_year": 2003, "base_score": 0.84},
    {"item_id": "7", "title": "Action A", "genre": ["Action"], "release_year": 2001, "base_score": 0.83},
    {"item_id": "8", "title": "Romance A", "genre": ["Romance"], "release_year": 2001, "base_score": 0.81},
]


VALID_DSL = {
    "meta": {"request_id": "req-1", "user_summary": "Likes comedy", "top_k": 4},
    "groups": [
        {"group_id": "comedy", "filter_expression": "genre == 'Comedy'"},
        {"group_id": "kids", "filter_expression": 'genre == "Children\'s"'},
        {"group_id": "horror", "filter_expression": "genre == 'Horror'"},
    ],
    "constraints": {
        "filters": [{"action": "exclude", "target_group": "horror"}],
        "quotas": [{"target_group": "kids", "min_count": 1}],
        "diversity": [{"attribute": "dominant_genre", "window_size": 3, "max_repetition": 1}],
    },
    "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "comedy", "weight": 0.2}]},
    "tie_break": ["base_score desc", "item_id asc"],
}


class RankDSLCoreTests(unittest.TestCase):
    def test_parser_accepts_valid_dsl(self):
        parsed = parse_ranking_dsl(VALID_DSL)
        self.assertEqual(parsed["meta"]["top_k"], 4)
        self.assertEqual(len(parsed["groups"]), 3)
        self.assertIn("ast", parsed["groups"][0])
        self.assertTrue(canonicalize_dsl(parsed).startswith("{"))

    def test_parser_accepts_rankdsl_lite_and_compiles_to_full_dsl(self):
        lite = {
            "top_k": 4,
            "filters": [{"field": "genre", "op": "==", "value": "Horror"}],
            "quotas": [{"field": "genre", "value": "Comedy", "min_count": 1}],
            "diversity": [{"field": "dominant_genre"}],
        }
        parsed = parse_ranking_dsl(lite)
        self.assertEqual(parsed["meta"]["top_k"], 4)
        self.assertEqual(len(parsed["groups"]), 2)
        self.assertEqual(parsed["constraints"]["filters"][0]["action"], "exclude")
        self.assertEqual(parsed["constraints"]["quotas"][0]["min_count"], 1)
        self.assertEqual(parsed["constraints"]["diversity"][0]["attribute"], "dominant_genre")

    def test_parser_rejects_unsupported_field(self):
        invalid = {
            **VALID_DSL,
            "groups": VALID_DSL["groups"] + [{"group_id": "bad", "filter_expression": "foo > 10"}],
        }
        result = verify_dsl(invalid, TOY_CANDIDATES)
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0]["code"], "unknown_field")
        self.assertEqual(result.errors[0]["error_type"], "metadata")

    def test_verifier_detects_infeasible_quota(self):
        infeasible = {
            **VALID_DSL,
            "constraints": {
                "filters": [],
                "quotas": [{"target_group": "horror", "min_count": 3}],
                "diversity": [],
            },
        }
        result = verify_dsl(infeasible, TOY_CANDIDATES)
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0]["code"], "infeasible_on_candidates")
        self.assertEqual(result.errors[0]["error_type"], "feasibility")
        self.assertIn("Relax filters or lower quota requirements", result.errors[0]["suggestion"])

    def test_verifier_surfaces_schema_errors_as_structured_payload(self):
        invalid = {
            **VALID_DSL,
            "tie_break": ["item_id asc", "base_score desc"],
        }
        result = verify_dsl(invalid, TOY_CANDIDATES)
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0]["code"], "schema_error")
        self.assertEqual(result.errors[0]["error_type"], "schema")
        self.assertIn("RankDSL schema", result.errors[0]["suggestion"])

    def test_greedy_solver_produces_feasible_ranking(self):
        parsed = parse_ranking_dsl(VALID_DSL)
        result = GreedySolver().solve(parsed, TOY_CANDIDATES)
        self.assertTrue(result.feasible)
        self.assertEqual(len(result.ranking), 4)
        self.assertTrue(all("Horror" not in candidate.genre for candidate in result.ranking))

    def test_ilp_solver_matches_constraints(self):
        parsed = parse_ranking_dsl(VALID_DSL)
        result = ILPSolver().solve(parsed, TOY_CANDIDATES)
        self.assertTrue(result.feasible)
        self.assertEqual(len(result.ranking), 4)
        self.assertTrue(any("Children's" in candidate.genre for candidate in result.ranking))

    def test_ilp_solver_falls_back_to_greedy_when_candidate_count_exceeds_threshold(self):
        parsed = parse_ranking_dsl(VALID_DSL)
        result = ILPSolver(max_candidates=5).solve(parsed, TOY_CANDIDATES)
        self.assertTrue(result.feasible)
        self.assertEqual(result.metadata["solver_requested"], "ilp_fallback_exact")
        self.assertEqual(result.metadata["solver_effective"], "greedy")
        self.assertEqual(result.metadata["fallback_reason"], "candidate_count>5")

    def test_stub_llm_compile_surfaces_valid_json(self):
        client = RankDSLLLMClient(mode="stub")
        response = client.compile_rankdsl(
            {
                "request_id": "req-2",
                "user_summary": "Likes comedy",
                "constraint_text": "Top-10 must contain at least 2 Children's titles and exclude Horror titles.",
            }
        )
        result = verify_dsl(response.text)
        self.assertTrue(result.ok)

    def test_verifier_accepts_rankdsl_lite_payload(self):
        result = verify_dsl(
            {
                "top_k": 4,
                "filters": [{"field": "genre", "op": "==", "value": "Horror"}],
                "quotas": [{"field": "genre", "value": "Comedy", "min_count": 1}],
                "diversity": [{"field": "dominant_genre"}],
            },
            TOY_CANDIDATES,
        )
        self.assertTrue(result.ok)

    def test_client_writes_parse_debug_log(self):
        with TemporaryDirectory() as tmpdir:
            parse_log_path = Path(tmpdir) / "parse_debug.jsonl"
            client = RankDSLLLMClient(mode="stub", parse_log_path=parse_log_path)
            payload = client.parse_json_response(
                'analysis first\n```json\n{"meta":{"top_k":1},"groups":[],"constraints":{"filters":[],"quotas":[],"diversity":[]},"objective":{"base_score_weight":1.0,"group_boosts":[]},"tie_break":["base_score desc","item_id asc"]}\n```',
                meta={"interaction_type": "compile_rankdsl", "request_id": "req-debug"},
            )
            self.assertEqual(payload["meta"]["top_k"], 1)
            rows = [line for line in parse_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            logged = json.loads(rows[0])
            self.assertTrue(logged["parse_ok"])
            self.assertEqual(logged["request_id"], "req-debug")
            self.assertIn("raw_preview", logged)
            self.assertIn("candidate_positions", logged)

    def test_parser_accepts_amazon_category_and_price_fields(self):
        amazon_dsl = {
            "meta": {"request_id": "req-amz", "user_summary": "Likes mystery", "top_k": 2},
            "groups": [
                {"group_id": "mystery", "filter_expression": "categories == 'Mystery'"},
                {"group_id": "expensive", "filter_expression": "price > 20"},
            ],
            "constraints": {
                "filters": [{"action": "exclude", "target_group": "expensive"}],
                "quotas": [{"target_group": "mystery", "min_count": 1}],
                "diversity": [{"attribute": "dominant_category", "window_size": 2, "max_repetition": 2}],
            },
            "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "mystery", "weight": 0.2}]},
            "tie_break": ["base_score desc", "item_id asc"],
        }
        result = verify_dsl(
            amazon_dsl,
            [
                {"item_id": "a", "title": "A", "categories": ["Mystery"], "price": 9.99, "base_score": 1.0},
                {"item_id": "b", "title": "B", "categories": ["Business"], "price": 25.0, "base_score": 0.9},
                {"item_id": "c", "title": "C", "categories": ["Mystery"], "price": 7.0, "base_score": 0.8},
            ],
        )
        self.assertTrue(result.ok)

    def test_runner_selects_hit_users_only_by_default(self):
        requests = [
            {"request_id": "a", "user_id": "u1", "target_item_id": "i1"},
            {"request_id": "b", "user_id": "u2", "target_item_id": "i9"},
        ]
        candidate_lookup = {
            "u1": {"user_id": "u1", "candidates": [{"item_id": "i1"}, {"item_id": "i2"}]},
            "u2": {"user_id": "u2", "candidates": [{"item_id": "i3"}, {"item_id": "i4"}]},
        }
        selected, info = select_requests_for_evaluation(
            requests,
            candidate_lookup,
            max_users=100,
            hit_users_only=True,
            sample_seed=2026,
        )
        self.assertEqual([row["request_id"] for row in selected], ["a"])
        self.assertEqual(info["filtered_miss_users"], 1)

    def test_runner_keeps_all_hit_users_when_max_eval_users_is_zero(self):
        requests = [
            {"request_id": "a", "user_id": "u1", "target_item_id": "i1"},
            {"request_id": "b", "user_id": "u2", "target_item_id": "i2"},
        ]
        candidate_lookup = {
            "u1": {"user_id": "u1", "candidates": [{"item_id": "i1"}, {"item_id": "i3"}]},
            "u2": {"user_id": "u2", "candidates": [{"item_id": "i2"}, {"item_id": "i4"}]},
        }
        selected, info = select_requests_for_evaluation(
            requests,
            candidate_lookup,
            max_users=0,
            hit_users_only=True,
            sample_seed=2026,
        )
        self.assertEqual([row["request_id"] for row in selected], ["a", "b"])
        self.assertEqual(info["eligible_total"], 2)

    def test_request_regeneration_detects_scenario_size_mismatch(self):
        with TemporaryDirectory() as tmpdir:
            requests_path = Path(tmpdir) / "requests.jsonl"
            write_jsonl(
                requests_path,
                [
                    {
                        "request_id": f"filter_horror-{index:03d}",
                        "dataset_name": "ml-1m",
                        "scenario_id": "filter_horror",
                        "user_id": f"u{index}",
                    }
                    for index in range(2)
                ],
            )
            should_regenerate, reason = needs_request_regeneration(
                "RankDSL/dataset/ml-1m",
                requests_path,
                scenario_size=50,
            )
            self.assertTrue(should_regenerate)
            self.assertIn("expected 300 requests", reason)

    def test_request_regeneration_detects_non_hit_requests_when_hit_users_only(self):
        with TemporaryDirectory() as tmpdir:
            requests_path = Path(tmpdir) / "requests.jsonl"
            write_jsonl(
                requests_path,
                [
                    {
                        "request_id": "filter_horror-000",
                        "dataset_name": "ml-1m",
                        "scenario_id": "filter_horror",
                        "user_id": "u1",
                        "target_item_id": "missing",
                    },
                    {
                        "request_id": "quota_comedy-000",
                        "dataset_name": "ml-1m",
                        "scenario_id": "quota_comedy",
                        "user_id": "u2",
                        "target_item_id": "i2",
                    },
                    {
                        "request_id": "quota_children-000",
                        "dataset_name": "ml-1m",
                        "scenario_id": "quota_children",
                        "user_id": "u3",
                        "target_item_id": "i3",
                    },
                    {
                        "request_id": "quota_comedy_filter_horror-000",
                        "dataset_name": "ml-1m",
                        "scenario_id": "quota_comedy_filter_horror",
                        "user_id": "u4",
                        "target_item_id": "i4",
                    },
                    {
                        "request_id": "diversity_dominant_genre-000",
                        "dataset_name": "ml-1m",
                        "scenario_id": "diversity_dominant_genre",
                        "user_id": "u5",
                        "target_item_id": "i5",
                    },
                    {
                        "request_id": "quota_children_diversity_filter_horror-000",
                        "dataset_name": "ml-1m",
                        "scenario_id": "quota_children_diversity_filter_horror",
                        "user_id": "u6",
                        "target_item_id": "i6",
                    },
                ],
            )
            should_regenerate, reason = needs_request_regeneration(
                "RankDSL/dataset/ml-1m",
                requests_path,
                scenario_size=1,
                candidate_lookup={
                    "u1": {"user_id": "u1", "candidates": [{"item_id": "i1"}]},
                    "u2": {"user_id": "u2", "candidates": [{"item_id": "i2"}]},
                    "u3": {"user_id": "u3", "candidates": [{"item_id": "i3"}]},
                    "u4": {"user_id": "u4", "candidates": [{"item_id": "i4"}]},
                    "u5": {"user_id": "u5", "candidates": [{"item_id": "i5"}]},
                    "u6": {"user_id": "u6", "candidates": [{"item_id": "i6"}]},
                },
                hit_users_only=True,
            )
            self.assertTrue(should_regenerate)
            self.assertIn("non-hit users", reason)

    def test_generate_requests_samples_from_hit_users_when_candidate_lookup_provided(self):
        class FakeReader:
            test_target_map = {"u1": "i1", "u2": "i2", "u3": "i3"}

            def eligible_users(self):
                return ["u1", "u2", "u3"]

            def build_user_summary(self, user_id):
                return f"summary-{user_id}"

            def get_user_profile(self, user_id):
                return f"profile-{user_id}"

            def render_history(self, user_id, max_events=20):
                return f"history-{user_id}"

        fake_spec = {
            "dataset_name": "ml-1m",
            "schema_fields": ["genre", "dominant_genre", "release_year"],
            "scenarios": [{"scenario_id": "filter_horror", "constraint_text": "Top-10 must exclude Horror titles."}],
            "reference_builder": lambda request_id, user_summary, scenario_id: VALID_DSL,
        }

        with patch("RankDSL.experiments.request_builder.get_dataset_name", return_value="ml-1m"), patch(
            "RankDSL.experiments.request_builder.get_dataset_spec", return_value=fake_spec
        ), patch("RankDSL.experiments.request_builder.get_reader", return_value=FakeReader()):
            requests = generate_requests(
                "RankDSL/dataset/ml-1m",
                scenario_size=2,
                seed=2026,
                candidate_lookup={
                    "u1": {"user_id": "u1", "candidates": [{"item_id": "i1"}]},
                    "u2": {"user_id": "u2", "candidates": [{"item_id": "i9"}]},
                    "u3": {"user_id": "u3", "candidates": [{"item_id": "i3"}]},
                },
                hit_users_only=True,
            )
        self.assertEqual(set(request["user_id"] for request in requests), {"u1", "u3"})

    def test_run_request_accepts_markdown_wrapped_rankdsl_json(self):
        class FencedJSONClient(RankDSLLLMClient):
            def __init__(self):
                super().__init__(mode="stub")

            def compile_rankdsl(self, request, paraphrase_index=0, repair_error=None):
                return LLMResponse(
                    text=f"```json\n{canonicalize_dsl(parse_ranking_dsl(VALID_DSL))}\n```",
                    model="fake",
                )

            def direct_rerank(self, request, candidates, paraphrase_index=0):
                ranking = [candidate["item_id"] for candidate in candidates]
                return LLMResponse(text=str(ranking).replace("'", '"'), model="fake")

        request = {
            "request_id": "req-3",
            "user_id": "u1",
            "scenario_id": "quota_children",
            "target_item_id": "3",
            "dataset_name": "ml-1m",
            "constraint_text": "Top-10 must contain at least 2 Children's titles.",
            "user_summary": "Likes family movies",
            "reference_dsl": VALID_DSL,
        }
        result = run_request(
            request,
            {"user_id": "u1", "candidates": TOY_CANDIDATES},
            FencedJSONClient(),
            verbose=False,
        )
        self.assertEqual(result["rankdsl"]["compile_success_rate"], 1.0)
        self.assertEqual(result["rankdsl"]["canonical_program_agreement"], 1.0)

    def test_run_request_uses_structured_repair_payload(self):
        class RepairAwareClient(RankDSLLLMClient):
            def __init__(self):
                super().__init__(mode="stub")
                self.repair_payloads = []

            def compile_rankdsl(self, request, paraphrase_index=0, repair_error=None):
                if repair_error is None:
                    return LLMResponse(text='{"meta": {"top_k": 10}, "groups": [}', model="fake")
                self.repair_payloads.append(repair_error)
                return LLMResponse(text=canonicalize_dsl(parse_ranking_dsl(VALID_DSL)), model="fake")

            def direct_rerank(self, request, candidates, paraphrase_index=0):
                ranking = [candidate["item_id"] for candidate in candidates]
                return LLMResponse(text=str(ranking).replace("'", '"'), model="fake")

        client = RepairAwareClient()
        request = {
            "request_id": "req-4",
            "user_id": "u1",
            "scenario_id": "quota_children",
            "target_item_id": "3",
            "dataset_name": "ml-1m",
            "constraint_text": "Top-10 must contain at least 2 Children's titles.",
            "user_summary": "Likes family movies",
            "reference_dsl": VALID_DSL,
        }
        result = run_request(
            request,
            {"user_id": "u1", "candidates": TOY_CANDIDATES},
            client,
            verbose=False,
        )
        self.assertEqual(result["rankdsl"]["compile_success_rate"], 1.0)
        self.assertEqual(len(client.repair_payloads), 3)
        self.assertTrue(all(payload.startswith('{"code"') for payload in client.repair_payloads))

    def test_client_loads_legacy_api_config_from_testapi(self):
        config = RankDSLLLMClient._load_legacy_api_config()
        self.assertIn("api_key", config)
        self.assertIn("base_url", config)
        self.assertTrue(config["api_key"])
        self.assertTrue(config["base_url"])


if __name__ == "__main__":
    unittest.main()
