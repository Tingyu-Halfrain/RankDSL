from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from .runtime import ALLOWED_DIVERSITY_ATTRIBUTES, ALLOWED_FILTER_FIELDS, RankDSLError


FILTER_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
    |(?P<lparen>\()
    |(?P<rparen>\))
    |(?P<op>==|!=|>=|<=|>|<)
    |(?P<number>\d+(?:\.\d+)?)
    |(?P<string>'(?:[^']|'(?=[A-Za-z]))*'|"(?:[^"\\]|\\.)*")
    |(?P<keyword>\band\b|\bor\b)
    |(?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<mismatch>.)
    """,
    re.VERBOSE,
)

DEFAULT_TIE_BREAK = ["base_score desc", "item_id asc"]


class FilterExpressionParser:
    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.index = 0

    def _tokenize(self, text: str) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        for match in FILTER_TOKEN_RE.finditer(text):
            kind = match.lastgroup
            value = match.group()
            if kind == "space":
                continue
            if kind == "mismatch":
                raise RankDSLError(
                    "syntax_error",
                    f"Unexpected token in filter expression: {value}",
                    {"token": value, "expression": text},
                )
            if kind == "keyword":
                value = value.lower()
            tokens.append((kind, value))
        return tokens

    def parse(self) -> Dict[str, Any]:
        if not self.tokens:
            raise RankDSLError("syntax_error", "Empty filter expression")
        expr = self._parse_or()
        if self.index != len(self.tokens):
            raise RankDSLError(
                "syntax_error",
                "Trailing tokens in filter expression",
                {"remaining_tokens": self.tokens[self.index :]},
            )
        return expr

    def _peek(self) -> Tuple[str, str] | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _consume(self, expected_kind: str | None = None, expected_value: str | None = None) -> Tuple[str, str]:
        token = self._peek()
        if token is None:
            raise RankDSLError("syntax_error", "Unexpected end of filter expression")
        kind, value = token
        if expected_kind and kind != expected_kind:
            raise RankDSLError(
                "syntax_error",
                f"Expected token kind {expected_kind}, got {kind}",
                {"token": token},
            )
        if expected_value and value != expected_value:
            raise RankDSLError(
                "syntax_error",
                f"Expected token {expected_value}, got {value}",
                {"token": token},
            )
        self.index += 1
        return token

    def _parse_or(self) -> Dict[str, Any]:
        node = self._parse_and()
        while self._peek() == ("keyword", "or"):
            self._consume("keyword", "or")
            node = {"type": "or", "left": node, "right": self._parse_and()}
        return node

    def _parse_and(self) -> Dict[str, Any]:
        node = self._parse_primary()
        while self._peek() == ("keyword", "and"):
            self._consume("keyword", "and")
            node = {"type": "and", "left": node, "right": self._parse_primary()}
        return node

    def _parse_primary(self) -> Dict[str, Any]:
        token = self._peek()
        if token is None:
            raise RankDSLError("syntax_error", "Unexpected end of filter expression")
        if token[0] == "lparen":
            self._consume("lparen")
            expr = self._parse_or()
            self._consume("rparen")
            return expr
        return self._parse_atom()

    def _parse_atom(self) -> Dict[str, Any]:
        _, field = self._consume("ident")
        _, op = self._consume("op")
        token = self._consume()
        if token[0] not in {"string", "number", "ident"}:
            raise RankDSLError(
                "syntax_error",
                "Filter value must be a string, number, or bare identifier",
                {"token": token},
            )
        value = self._coerce_literal(token)
        return {"type": "atom", "field": field, "op": op, "value": value}

    @staticmethod
    def _coerce_literal(token: Tuple[str, str]) -> Any:
        kind, value = token
        if kind == "string":
            return value[1:-1]
        if kind == "number":
            return int(value) if "." not in value else float(value)
        return value


def parse_filter_expression(expression: str) -> Dict[str, Any]:
    parser = FilterExpressionParser(expression)
    ast = parser.parse()
    validate_filter_ast(ast)
    return ast


def validate_filter_ast(ast: Dict[str, Any]) -> None:
    node_type = ast["type"]
    if node_type in {"and", "or"}:
        validate_filter_ast(ast["left"])
        validate_filter_ast(ast["right"])
        return
    if node_type != "atom":
        raise RankDSLError("syntax_error", f"Unknown AST node type: {node_type}", {"node_type": node_type})

    field = ast["field"]
    if field not in ALLOWED_FILTER_FIELDS:
        raise RankDSLError("unknown_field", f"Unsupported field in filter expression: {field}", {"field": field})

    op = ast["op"]
    if field in {"genre", "categories"}:
        if op not in {"==", "!="}:
            raise RankDSLError(
                "unsupported_operator",
                f"Field {field} only supports == and !=",
                {"field": field, "operator": op},
            )
    elif op not in {"==", "!=", ">", ">=", "<", "<="}:
        raise RankDSLError(
            "unsupported_operator",
            f"Unsupported operator {op}",
            {"field": field, "operator": op},
        )


def parse_ranking_dsl(payload: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RankDSLError("syntax_error", f"Invalid JSON DSL: {exc.msg}", {"position": exc.pos}) from exc
    else:
        raw = payload

    if not isinstance(raw, dict):
        raise RankDSLError("syntax_error", "DSL payload must be a JSON object")

    meta = raw.get("meta") or {}
    top_k = int(meta.get("top_k", 10))
    if top_k <= 0:
        raise RankDSLError("invalid_topk", "top_k must be positive", {"top_k": top_k})

    groups: List[Dict[str, Any]] = []
    seen_group_ids = set()
    for group in raw.get("groups", []):
        group_id = str(group["group_id"])
        if group_id in seen_group_ids:
            raise RankDSLError("syntax_error", f"Duplicate group id: {group_id}", {"group_id": group_id})
        seen_group_ids.add(group_id)
        expression = str(group["filter_expression"])
        groups.append(
            {"group_id": group_id, "filter_expression": expression, "ast": parse_filter_expression(expression)}
        )

    constraints = raw.get("constraints") or {}
    filters = []
    for filter_rule in constraints.get("filters", []):
        action = filter_rule.get("action", "exclude")
        if action != "exclude":
            raise RankDSLError("unsupported_operator", "Only exclude filters are supported", {"action": action})
        filters.append({"action": action, "target_group": str(filter_rule["target_group"])})

    quotas = []
    for quota in constraints.get("quotas", []):
        min_count = int(quota.get("min_count", 0))
        max_count = quota.get("max_count")
        if max_count is not None:
            max_count = int(max_count)
            if max_count < min_count:
                raise RankDSLError(
                    "syntax_error",
                    "Quota max_count must be >= min_count",
                    {"quota": quota},
                )
        quotas.append(
            {
                "target_group": str(quota["target_group"]),
                "min_count": min_count,
                "max_count": max_count,
            }
        )

    diversity = []
    for rule in constraints.get("diversity", []):
        attribute = str(rule["attribute"])
        if attribute not in ALLOWED_DIVERSITY_ATTRIBUTES:
            raise RankDSLError(
                "unknown_field",
                f"Unsupported diversity attribute: {attribute}",
                {"attribute": attribute},
            )
        window_size = int(rule["window_size"])
        max_repetition = int(rule["max_repetition"])
        if window_size <= 0 or max_repetition <= 0:
            raise RankDSLError("syntax_error", "window_size and max_repetition must be positive", {"rule": rule})
        diversity.append(
            {
                "attribute": attribute,
                "window_size": window_size,
                "max_repetition": max_repetition,
            }
        )

    objective_raw = raw.get("objective") or {}
    objective = {
        "base_score_weight": float(objective_raw.get("base_score_weight", 1.0)),
        "group_boosts": [],
    }
    for boost in objective_raw.get("group_boosts", []):
        objective["group_boosts"].append(
            {"target_group": str(boost["target_group"]), "weight": float(boost.get("weight", 0.0))}
        )

    tie_break = raw.get("tie_break") or list(DEFAULT_TIE_BREAK)
    if tie_break != DEFAULT_TIE_BREAK:
        raise RankDSLError(
            "unsupported_operator",
            "tie_break is fixed to ['base_score desc', 'item_id asc'] in v1",
            {"tie_break": tie_break},
        )

    dsl = {
        "meta": {
            "request_id": str(meta.get("request_id", "")),
            "user_summary": str(meta.get("user_summary", "")),
            "top_k": top_k,
        },
        "groups": groups,
        "constraints": {"filters": filters, "quotas": quotas, "diversity": diversity},
        "objective": objective,
        "tie_break": list(DEFAULT_TIE_BREAK),
    }
    validate_group_references(dsl)
    return dsl


def validate_group_references(dsl: Dict[str, Any]) -> None:
    group_ids = {group["group_id"] for group in dsl["groups"]}
    for container in ("filters", "quotas"):
        for rule in dsl["constraints"][container]:
            target_group = rule["target_group"]
            if target_group not in group_ids:
                raise RankDSLError(
                    "syntax_error",
                    f"Constraint references undefined group: {target_group}",
                    {"target_group": target_group},
                )
    for boost in dsl["objective"]["group_boosts"]:
        if boost["target_group"] not in group_ids:
            raise RankDSLError(
                "syntax_error",
                f"Boost references undefined group: {boost['target_group']}",
                {"target_group": boost["target_group"]},
            )


def canonicalize_dsl(dsl: Dict[str, Any]) -> str:
    serializable = {
        "meta": dsl["meta"],
        "groups": [
            {"group_id": group["group_id"], "filter_expression": group["filter_expression"]} for group in dsl["groups"]
        ],
        "constraints": dsl["constraints"],
        "objective": dsl["objective"],
        "tie_break": dsl["tie_break"],
    }
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dump_example_schema() -> Dict[str, Any]:
    return {
        "meta": {"request_id": "req-001", "user_summary": "Likes Comedy and Animation", "top_k": 10},
        "groups": [
            {"group_id": "comedy", "filter_expression": "genre == 'Comedy'"},
            {"group_id": "horror", "filter_expression": "genre == 'Horror'"},
        ],
        "constraints": {
            "filters": [{"action": "exclude", "target_group": "horror"}],
            "quotas": [{"target_group": "comedy", "min_count": 3}],
            "diversity": [{"attribute": "dominant_genre", "window_size": 3, "max_repetition": 1}],
        },
        "objective": {"base_score_weight": 1.0, "group_boosts": [{"target_group": "comedy", "weight": 0.2}]},
        "tie_break": list(DEFAULT_TIE_BREAK),
    }
