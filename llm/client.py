from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .prompts import build_direct_rerank_messages, build_rankdsl_messages

try:
    from openai import APIConnectionError, OpenAI
except ImportError:  # pragma: no cover
    APIConnectionError = None
    OpenAI = None


CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
LEGACY_OPENAI_CONFIG_RE = re.compile(
    r"OpenAI\s*\(\s*.*?api_key\s*=\s*['\"](?P<api_key>[^'\"]+)['\"].*?"
    r"base_url\s*=\s*['\"](?P<base_url>[^'\"]+)['\"]",
    re.DOTALL,
)


@dataclass
class LLMResponse:
    text: str
    model: str


class RankDSLLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "claude-opus-4-6",
        mode: Optional[str] = None,
        log_path: str | Path | None = None,
    ):
        self.mode = mode or os.environ.get("RANKDSL_LLM_MODE", "stub")
        self.model = model
        fallback_config = self._load_legacy_api_config()
        self.api_key = api_key or os.environ.get("RANKDSL_API_KEY") or fallback_config.get("api_key")
        self.base_url = base_url or os.environ.get("RANKDSL_BASE_URL") or fallback_config.get("base_url")
        self.log_path = Path(log_path) if log_path else None
        self._client = None
        if self.mode == "api":
            if OpenAI is None:
                raise RuntimeError("openai package is required for api mode")
            if not self.api_key or not self.base_url:
                raise RuntimeError(
                    "API config missing for api mode. Set RANKDSL_API_KEY and RANKDSL_BASE_URL, "
                    "or pass --api-key/--base-url."
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _append_log(self, payload: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _log_interaction(
        self,
        messages: List[Dict[str, str]],
        response_text: str | None = None,
        meta: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "model": self.model,
            "messages": messages,
        }
        if meta:
            payload.update(meta)
        if response_text is not None:
            payload["response_text"] = response_text
        if error is not None:
            payload["error"] = error
        self._append_log(payload)

    def _chat(self, messages: List[Dict[str, str]], meta: Dict[str, Any] | None = None) -> LLMResponse:
        if self.mode != "api":
            raise RuntimeError("Direct API calls are only available in api mode")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                messages=messages,
            )
        except Exception as exc:
            self._log_interaction(messages, meta=meta, error=repr(exc))
            if APIConnectionError is not None and isinstance(exc, APIConnectionError):
                raise RuntimeError(
                    f"Failed to reach API endpoint {self.base_url}. "
                    "Check network/DNS connectivity and verify the base_url is reachable."
                ) from exc
            raise
        response_text = response.choices[0].message.content
        self._log_interaction(messages, response_text=response_text, meta=meta)
        return LLMResponse(text=response_text, model=self.model)

    @staticmethod
    def _extract_json_block(text: str) -> str:
        stripped = text.strip()
        fenced = CODE_FENCE_RE.match(stripped)
        if fenced:
            stripped = fenced.group(1).strip()

        decoder = json.JSONDecoder()
        candidate_positions = [
            index for index, char in enumerate(stripped) if char in {"{", "["}
        ]
        for start in candidate_positions:
            try:
                _, end = decoder.raw_decode(stripped[start:])
                return stripped[start : start + end]
            except json.JSONDecodeError:
                continue
        raise ValueError("No valid JSON block found in LLM response")

    @staticmethod
    def _load_legacy_api_config() -> Dict[str, str]:
        legacy_path = Path(__file__).resolve().parents[2] / "AIresearcher" / "testAPI.py"
        if not legacy_path.exists():
            return {}
        try:
            content = legacy_path.read_text(encoding="utf-8")
        except OSError:
            return {}
        match = LEGACY_OPENAI_CONFIG_RE.search(content)
        if not match:
            return {}
        return {
            "api_key": match.group("api_key"),
            "base_url": match.group("base_url"),
        }

    def compile_rankdsl(
        self,
        request: Dict[str, str],
        paraphrase_index: int = 0,
        repair_error: str | None = None,
    ) -> LLMResponse:
        messages = build_rankdsl_messages(request, paraphrase_index, repair_error)
        meta = {
            "interaction_type": "compile_rankdsl",
            "request_id": request.get("request_id"),
            "user_id": request.get("user_id"),
            "scenario_id": request.get("scenario_id"),
            "paraphrase_index": paraphrase_index,
            "is_repair": repair_error is not None,
        }
        if self.mode == "stub":
            response_text = json.dumps(self._stub_compile(request), ensure_ascii=False)
            self._log_interaction(messages, response_text=response_text, meta=meta)
            return LLMResponse(text=response_text, model="stub")
        return self._chat(messages, meta=meta)

    def direct_rerank(
        self,
        request: Dict[str, str],
        candidates: Sequence[Dict[str, str]],
        paraphrase_index: int = 0,
    ) -> LLMResponse:
        messages = build_direct_rerank_messages(request, candidates, paraphrase_index)
        meta = {
            "interaction_type": "direct_rerank",
            "request_id": request.get("request_id"),
            "user_id": request.get("user_id"),
            "scenario_id": request.get("scenario_id"),
            "paraphrase_index": paraphrase_index,
            "candidate_count": len(candidates),
        }
        if self.mode == "stub":
            ranking = [candidate["item_id"] for candidate in candidates[:10]]
            response_text = json.dumps(ranking, ensure_ascii=False)
            self._log_interaction(messages, response_text=response_text, meta=meta)
            return LLMResponse(text=response_text, model="stub")
        return self._chat(messages, meta=meta)

    def parse_json_response(self, text: str):
        return json.loads(self._extract_json_block(text))

    @staticmethod
    def _stub_compile(request: Dict[str, str]) -> Dict[str, object]:
        if "reference_dsl" in request:
            return request["reference_dsl"]
        constraint_text = request["constraint_text"].lower()
        groups = []
        filters = []
        quotas = []
        diversity = []
        boosts = []

        def ensure_group(group_id: str, expression: str) -> None:
            if not any(group["group_id"] == group_id for group in groups):
                groups.append({"group_id": group_id, "filter_expression": expression})

        if "horror" in constraint_text:
            ensure_group("horror", "genre == 'Horror'")
            filters.append({"action": "exclude", "target_group": "horror"})
        if "comedy" in constraint_text:
            ensure_group("comedy", "genre == 'Comedy'")
            quotas.append({"target_group": "comedy", "min_count": 3})
            boosts.append({"target_group": "comedy", "weight": 0.2})
        if "children" in constraint_text:
            ensure_group("children", 'genre == "Children\'s"')
            quotas.append({"target_group": "children", "min_count": 2})
            boosts.append({"target_group": "children", "weight": 0.2})
        if "window" in constraint_text or "不重复" in constraint_text or "diversity" in constraint_text:
            diversity.append({"attribute": "dominant_genre", "window_size": 3, "max_repetition": 1})

        return {
            "meta": {
                "request_id": request["request_id"],
                "user_summary": request["user_summary"],
                "top_k": 10,
            },
            "groups": groups,
            "constraints": {"filters": filters, "quotas": quotas, "diversity": diversity},
            "objective": {"base_score_weight": 1.0, "group_boosts": boosts},
            "tie_break": ["base_score desc", "item_id asc"],
        }
