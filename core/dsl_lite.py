from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .runtime import ALLOWED_DIVERSITY_ATTRIBUTES, ALLOWED_FILTER_FIELDS


STRING_FILTER_FIELDS = {"genre", "categories", "dominant_genre", "dominant_category", "brand"}
NUMERIC_FILTER_FIELDS = {"release_year", "price"}
LITE_ALLOWED_OPERATORS = {"==", "!=", ">", ">=", "<", "<="}
LITE_DEFAULT_TIE_BREAK = ["base_score desc", "item_id asc"]
LITE_DEFAULT_BOOST = 0.2


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "value"


def _literal(value: Any) -> str:
    if isinstance(value, str):
        if "'" in value and '"' not in value:
            return f'"{value}"'
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


class LiteFilterRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    op: str = "=="
    value: str | int | float

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in ALLOWED_FILTER_FIELDS:
            raise ValueError(f"Unsupported filter field: {value}")
        return value

    @field_validator("op")
    @classmethod
    def validate_operator(cls, value: str) -> str:
        if value not in LITE_ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {value}")
        return value

    @model_validator(mode="after")
    def validate_field_operator(self) -> "LiteFilterRule":
        if self.field in {"genre", "categories"} and self.op not in {"==", "!="}:
            raise ValueError(f"Field {self.field} only supports == and !=")
        if self.field in STRING_FILTER_FIELDS and self.field not in {"genre", "categories"} and self.op not in {"==", "!="}:
            raise ValueError(f"Field {self.field} only supports == and !=")
        if self.field in NUMERIC_FILTER_FIELDS and isinstance(self.value, str):
            raise ValueError(f"Field {self.field} requires a numeric value")
        return self


class LiteQuotaRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str
    min_count: int = Field(ge=0)

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in {"genre", "categories"}:
            raise ValueError("Quota field must be genre or categories")
        return value


class LiteDiversityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    window_size: int = Field(default=3, gt=0)
    max_repetition: int = Field(default=1, gt=0)

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if value not in ALLOWED_DIVERSITY_ATTRIBUTES:
            raise ValueError(f"Unsupported diversity field: {value}")
        return value


class RankDSLLiteProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, gt=0)
    filters: List[LiteFilterRule] = Field(default_factory=list)
    quotas: List[LiteQuotaRule] = Field(default_factory=list)
    diversity: List[LiteDiversityRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_non_empty(self) -> "RankDSLLiteProgram":
        if not self.filters and not self.quotas and not self.diversity:
            raise ValueError("At least one of filters, quotas, or diversity must be provided")
        return self


def is_rankdsl_lite_payload(raw: Dict[str, Any]) -> bool:
    if not isinstance(raw, dict):
        return False
    if "groups" in raw or "constraints" in raw or "objective" in raw or "meta" in raw:
        return False
    return any(key in raw for key in ("filters", "quotas", "diversity", "top_k"))


def compile_rankdsl_lite(payload: Dict[str, Any], request_id: str = "", user_summary: str = "") -> Dict[str, Any]:
    program = RankDSLLiteProgram.model_validate(payload)

    groups: List[Dict[str, str]] = []
    filters: List[Dict[str, str]] = []
    quotas: List[Dict[str, Any]] = []
    boosts: List[Dict[str, Any]] = []
    group_index: Dict[tuple[str, str, Any], str] = {}

    def ensure_group(field: str, op: str, value: Any, prefix: str) -> str:
        key = (field, op, value)
        if key in group_index:
            return group_index[key]
        group_id = f"{prefix}_{field}_{_slug(op)}_{_slug(value)}"
        group_index[key] = group_id
        groups.append(
            {
                "group_id": group_id,
                "filter_expression": f"{field} {op} {_literal(value)}",
            }
        )
        return group_id

    for rule in program.filters:
        group_id = ensure_group(rule.field, rule.op, rule.value, "filter")
        filters.append({"action": "exclude", "target_group": group_id})

    for rule in program.quotas:
        group_id = ensure_group(rule.field, "==", rule.value, "quota")
        quotas.append({"target_group": group_id, "min_count": rule.min_count})
        boosts.append({"target_group": group_id, "weight": LITE_DEFAULT_BOOST})

    diversity = [
        {
            "attribute": rule.field,
            "window_size": rule.window_size,
            "max_repetition": rule.max_repetition,
        }
        for rule in program.diversity
    ]

    return {
        "meta": {
            "request_id": request_id,
            "user_summary": user_summary,
            "top_k": program.top_k,
        },
        "groups": groups,
        "constraints": {
            "filters": filters,
            "quotas": quotas,
            "diversity": diversity,
        },
        "objective": {
            "base_score_weight": 1.0,
            "group_boosts": boosts,
        },
        "tie_break": list(LITE_DEFAULT_TIE_BREAK),
    }


def dump_example_lite_schema() -> Dict[str, Any]:
    return {
        "top_k": 10,
        "filters": [{"field": "genre", "op": "==", "value": "Horror"}],
        "quotas": [{"field": "genre", "value": "Comedy", "min_count": 3}],
        "diversity": [{"field": "dominant_genre", "window_size": 3, "max_repetition": 1}],
    }


def rankdsl_lite_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rankdsl_lite_program",
            "strict": True,
            "schema": RankDSLLiteProgram.model_json_schema(),
        },
    }
