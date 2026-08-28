"""Validation and table extraction for future real experiment results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import OmegaConf


def _invalid_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    if isinstance(value, str) and any(
        marker in value.upper()
        for marker in ("RESEARCH_OWNER_DECISION_REQUIRED", "REPLACE_WITH", "TEMPLATE")
    ):
        return True
    if isinstance(value, Mapping):
        return any(_invalid_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_invalid_value(item) for item in value)
    return False


def load_result_schemas(path: str | Path) -> dict[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("result schema registry is invalid")
    families = raw.get("families")
    if not isinstance(families, dict):
        raise ValueError("result schema families are missing")
    return families


def validate_result(result: Mapping[str, Any], schemas: Mapping[str, Any]) -> None:
    if result.get("template_only") is True:
        raise ValueError("result template is not a formal result")
    family = result.get("result_family")
    if family not in schemas:
        raise ValueError(f"unknown result family: {family}")
    schema = schemas[str(family)]
    missing = [name for name in schema["required"] if name not in result]
    if missing:
        raise ValueError(f"result is missing required fields: {missing}")
    authority = result.get("authority_sha256")
    if not isinstance(authority, str) or len(authority) != 64:
        raise ValueError("formal result requires authority_sha256")
    if _invalid_value(result):
        raise ValueError(
            "formal result contains null, non-finite, or placeholder values"
        )
    metric_names = schema.get("metrics", [])
    if metric_names:
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("result metrics must be a mapping")
        missing_metrics = [name for name in metric_names if name not in metrics]
        if missing_metrics:
            raise ValueError(f"result is missing metrics: {missing_metrics}")
    for collection_name, fields_name in (
        ("rows", "row_fields"),
        ("amortization", "amortization_fields"),
    ):
        required_fields = schema.get(fields_name)
        if required_fields:
            rows = result.get(collection_name)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{collection_name} must be a non-empty list")
            for row in rows:
                if not isinstance(row, Mapping) or any(
                    field not in row for field in required_fields
                ):
                    raise ValueError(f"{collection_name} row is incomplete")


def collect_tables(
    result_paths: Iterable[str | Path], schema_path: str | Path
) -> dict[str, list[dict[str, Any]]]:
    schemas = load_result_schemas(schema_path)
    tables: dict[str, list[dict[str, Any]]] = {
        "accuracy_cost": [],
        "oracle_headroom": [],
        "router_training": [],
        "cache_savings": [],
        "generalization": [],
    }
    for raw_path in result_paths:
        path = Path(raw_path)
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_result(result, schemas)
        family = result["result_family"]
        if family == "evaluation":
            tables["accuracy_cost"].extend(result["rows"])
        elif family == "oracle":
            tables["oracle_headroom"].append(result["metrics"])
        elif family == "training":
            tables["router_training"].append(result["metrics"])
        elif family == "efficiency":
            tables["cache_savings"].extend(result["amortization"])
        elif family == "generalization":
            tables["generalization"].extend(result["rows"])
    return tables
