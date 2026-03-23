from __future__ import annotations

import json
from typing import Dict, List, Sequence

from ..core.dsl_parser import dump_example_schema


RANKDSL_PARAPHRASES = [
    "Compile the user request into a single valid RankDSL JSON program.",
    "Return only a RankDSL JSON object that encodes the request under the allowed schema.",
    "Act as a deterministic compiler. Output only one valid RankDSL JSON program.",
]

DIRECT_RERANK_PARAPHRASES = [
    "Return the best top-10 ranking as a JSON array of item_id strings.",
    "Output only a JSON array with 10 item_id values ordered from best to worst.",
    "Rank the candidates and return a JSON array of item_id strings, no explanation.",
]


def build_rankdsl_messages(
    request: Dict[str, str],
    paraphrase_index: int,
    repair_error: str | None = None,
) -> List[Dict[str, str]]:
    schema_example = json.dumps(dump_example_schema(), ensure_ascii=False, indent=2)
    allowed_fields = ", ".join(request.get("schema_fields", ["genre", "dominant_genre", "release_year"]))
    system_prompt = (
        "You are RankDSLCompiler.\n"
        f"Allowed fields in filter_expression: {allowed_fields}.\n"
        "Allowed boolean operators: and, or.\n"
        "Allowed filter action: exclude.\n"
        "Allowed output format: one JSON object only.\n"
        'The first character of your response must be "{", and the last character must be "}".\n'
        'Tie break must be ["base_score desc", "item_id asc"].\n'
        "Do not output markdown fences, prose, explanations, or extra text.\n"
        f"Reference schema example:\n{schema_example}"
    )
    user_prompt = (
        f"{RANKDSL_PARAPHRASES[paraphrase_index % len(RANKDSL_PARAPHRASES)]}\n"
        f"request_id: {request['request_id']}\n"
        f"user_profile: {request.get('user_profile', '')}\n"
        f"user_summary: {request['user_summary']}\n"
        f"user_history:\n{request.get('history_text', '')}\n"
        f"constraint_text: {request['constraint_text']}\n"
        "top_k: 10"
    )
    if repair_error:
        user_prompt += (
            "\nPrevious output failed verification."
            f"\nError: {repair_error}"
            "\nReturn one corrected JSON object only. Do not wrap it in ```json fences."
        )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_direct_rerank_messages(
    request: Dict[str, str],
    candidates: Sequence[Dict[str, str]],
    paraphrase_index: int,
) -> List[Dict[str, str]]:
    serialized_candidates = json.dumps(candidates, ensure_ascii=False, indent=2)
    system_prompt = (
        "You are a listwise reranker.\n"
        "Return only a JSON array of exactly 10 item_id strings.\n"
        "Use the user preference and candidate summaries."
    )
    user_prompt = (
        f"{DIRECT_RERANK_PARAPHRASES[paraphrase_index % len(DIRECT_RERANK_PARAPHRASES)]}\n"
        f"user_profile: {request.get('user_profile', '')}\n"
        f"user_summary: {request['user_summary']}\n"
        f"user_history:\n{request.get('history_text', '')}\n"
        f"constraint_text: {request['constraint_text']}\n"
        f"candidates:\n{serialized_candidates}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
