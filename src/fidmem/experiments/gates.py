"""Frozen, deterministic gate evaluators for paper experiments."""

from __future__ import annotations

from typing import Any, Mapping


def _placeholder(value: object) -> bool:
    return isinstance(value, str) and "RESEARCH_OWNER_DECISION_REQUIRED" in value


def evaluate_gate(
    gate_id: str, result: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> tuple[str, dict[str, bool]]:
    if any(_placeholder(value) for value in thresholds.values()):
        return "FAIL", {"thresholds_frozen": False}
    if gate_id == "environment_ready":
        checks = {
            "environment_valid": bool(result.get("environment_valid", False)),
            "source_identity_recorded": bool(
                result.get("source_identity_recorded", False)
            ),
        }
    elif gate_id == "dataset_frozen":
        checks = {
            name: bool(result.get(name, False))
            for name in ("manifests_complete", "video_disjoint", "hashes_valid")
        }
    elif gate_id == "authority_sealed":
        checks = {
            name: bool(result.get(name, False))
            for name in ("sealed", "authority_valid", "runtime_matches")
        }
    elif gate_id == "production_canary":
        checks = {
            name: bool(result.get(name, False))
            for name in (
                "provenance_passed",
                "cost_reconciliation_passed",
                "cache_isolation_passed",
                "resume_passed",
                "identity_consistency_passed",
                "namespace_isolation_passed",
                "manifest_complete",
                "collision_free",
            )
        }
    elif gate_id == "oracle_viability":
        checks = {
            "minimum_oracle_headroom": float(result.get("oracle_headroom", -1.0))
            >= float(thresholds["minimum_oracle_headroom"]),
            "maximum_missing_observation_rate": float(
                result.get("missing_observation_rate", 1.0)
            )
            <= float(thresholds["maximum_missing_observation_rate"]),
            "cost_projection_within_cap": bool(
                result.get("cost_projection_within_cap", False)
            ),
            "trajectory_records_complete": bool(
                result.get("trajectory_records_complete", False)
            ),
        }
    elif gate_id == "answerer_stability":
        maximum = float(thresholds["maximum_flip_rate"])
        checks = {
            "answer_flip_rate": float(result.get("answer_flip_rate", 1.0)) <= maximum,
            "label_flip_rate": float(result.get("label_flip_rate", 1.0)) <= maximum,
            "repeats_complete": bool(result.get("repeats_complete", False)),
        }
    elif gate_id == "zero_leakage":
        checks = {
            "confirmed_leakage_zero": int(result.get("confirmed_leakage", -1)) == 0,
            "video_level_isolation": bool(result.get("video_level_isolation", False)),
        }
    elif gate_id == "label_audit":
        checks = {
            "label_audit_passed": bool(result.get("label_audit_passed", False)),
            "audited_count_satisfied": int(result.get("audited_count", 0))
            >= int(thresholds["minimum_audited_count"]),
        }
    elif gate_id == "beam_reliability":
        checks = {
            "main_labels_unchanged": bool(result.get("main_labels_unchanged", False)),
            "audit_size_satisfied": int(result.get("audit_size", 0))
            >= int(thresholds["minimum_audit_size"]),
        }
    elif gate_id == "gist_recall":
        checks = {
            "top_k_recall": float(result.get("top_k_recall", -1.0))
            >= float(thresholds["minimum_top_k_recall"])
        }
    elif gate_id == "fixed_baselines":
        checks = {
            name: bool(result.get(name, False))
            for name in ("fixed_matrix_complete", "shared_identity_passed")
        }
    elif gate_id == "baseline_ready":
        checks = {
            name: bool(result.get(name, False))
            for name in (
                "all_fixed_complete",
                "all_adaptive_complete",
                "shared_identity_passed",
            )
        }
    elif gate_id == "bc_router":
        checks = {
            "bc_beats_rule": bool(result.get("bc_beats_rule", False)),
            "three_seeds_complete": int(result.get("completed_seeds", 0)) == 3,
            "frozen_observations_only": bool(
                result.get("frozen_observations_only", False)
            ),
        }
    elif gate_id == "bc_evaluation":
        checks = {
            name: bool(result.get(name, False))
            for name in ("evaluation_complete", "shared_identity_passed")
        }
    elif gate_id == "dagger":
        checks = {
            "minimum_rounds_complete": int(result.get("rounds_complete", 0))
            >= int(thresholds["minimum_rounds"]),
            "utility_gain": float(result.get("utility_gain", float("-inf")))
            >= float(thresholds["utility_gain_threshold"]),
            "regret_improvement_ratio": float(
                result.get("regret_improvement_ratio", float("-inf"))
            )
            >= float(thresholds["regret_improvement_ratio"]),
        }
    elif gate_id == "main_benchmark":
        checks = {
            "qualifying_benchmarks": int(result.get("qualifying_benchmarks", 0))
            >= int(thresholds["minimum_qualifying_benchmarks"]),
            "cost_or_accuracy_claim": bool(
                result.get("cost_or_accuracy_claim_satisfied", False)
            ),
            "all_seeds_complete": bool(result.get("all_seeds_complete", False)),
        }
    elif gate_id in {
        "ablations_complete",
        "efficiency_complete",
        "cross_dataset",
        "generalization_complete",
    }:
        checks = {
            name: bool(result.get(name, False))
            for name in ("planned_matrix_complete", "provenance_complete")
        }
    else:
        raise ValueError(f"no evaluator registered for gate {gate_id}")
    return ("PASS" if all(checks.values()) else "FAIL", checks)
