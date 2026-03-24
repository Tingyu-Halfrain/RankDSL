from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .runtime import ALLOWED_DIVERSITY_ATTRIBUTES


FILTER_EXPRESSION_PATTERN = r"^[A-Za-z0-9_.,:/\s=><!&|()'\"-]+$"
SCHEMA_DEFAULT_TIE_BREAK = ["base_score desc", "item_id asc"]


class MetaBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str = ""
    user_summary: str = ""
    top_k: int = Field(default=10, gt=0)


class GroupDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(min_length=1)
    filter_expression: str = Field(min_length=1, pattern=FILTER_EXPRESSION_PATTERN)


class FilterRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = "exclude"
    target_group: str = Field(min_length=1)

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        if value != "exclude":
            raise ValueError("Only exclude filters are supported")
        return value


class QuotaRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_group: str = Field(min_length=1)
    min_count: int = Field(default=0, ge=0)
    max_count: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "QuotaRule":
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("Quota max_count must be >= min_count")
        return self


class DiversityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: str
    window_size: int = Field(gt=0)
    max_repetition: int = Field(gt=0)

    @field_validator("attribute")
    @classmethod
    def validate_attribute(cls, value: str) -> str:
        if value not in ALLOWED_DIVERSITY_ATTRIBUTES:
            raise ValueError(f"Unsupported diversity attribute: {value}")
        return value


class ConstraintsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: List[FilterRule] = Field(default_factory=list)
    quotas: List[QuotaRule] = Field(default_factory=list)
    diversity: List[DiversityRule] = Field(default_factory=list)


class GroupBoost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_group: str = Field(min_length=1)
    weight: float = 0.0


class ObjectiveBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_score_weight: float = 1.0
    group_boosts: List[GroupBoost] = Field(default_factory=list)


class DSLProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: MetaBlock = Field(default_factory=MetaBlock)
    groups: List[GroupDefinition] = Field(default_factory=list)
    constraints: ConstraintsBlock = Field(default_factory=ConstraintsBlock)
    objective: ObjectiveBlock = Field(default_factory=ObjectiveBlock)
    tie_break: List[str] = Field(default_factory=lambda: list(SCHEMA_DEFAULT_TIE_BREAK))

    @field_validator("tie_break")
    @classmethod
    def validate_tie_break(cls, value: List[str]) -> List[str]:
        if value != SCHEMA_DEFAULT_TIE_BREAK:
            raise ValueError("tie_break must equal ['base_score desc', 'item_id asc']")
        return value

    @model_validator(mode="after")
    def validate_group_ids(self) -> "DSLProgram":
        seen = set()
        for group in self.groups:
            if group.group_id in seen:
                raise ValueError(f"Duplicate group id: {group.group_id}")
            seen.add(group.group_id)

        for rule in self.constraints.filters:
            if rule.target_group not in seen:
                raise ValueError(f"Constraint references undefined group: {rule.target_group}")
        for rule in self.constraints.quotas:
            if rule.target_group not in seen:
                raise ValueError(f"Constraint references undefined group: {rule.target_group}")
        for boost in self.objective.group_boosts:
            if boost.target_group not in seen:
                raise ValueError(f"Boost references undefined group: {boost.target_group}")
        return self


def rankdsl_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rankdsl_program",
            "strict": True,
            "schema": DSLProgram.model_json_schema(),
        },
    }
