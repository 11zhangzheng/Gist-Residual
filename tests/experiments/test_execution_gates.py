from fidmem.experiments.gates import evaluate_gate


def test_canary_requires_all_production_integrity_checks() -> None:
    result = {
        "provenance_passed": True,
        "cost_reconciliation_passed": True,
        "cache_isolation_passed": True,
        "resume_passed": True,
        "identity_consistency_passed": True,
        "namespace_isolation_passed": True,
        "manifest_complete": True,
        "collision_free": True,
    }
    assert evaluate_gate("production_canary", result, {})[0] == "PASS"
    result["resume_passed"] = False
    assert evaluate_gate("production_canary", result, {})[0] == "FAIL"


def test_unfrozen_oracle_threshold_cannot_pass() -> None:
    status, checks = evaluate_gate(
        "oracle_viability",
        {
            "oracle_headroom": 1.0,
            "missing_observation_rate": 0.0,
            "cost_projection_within_cap": True,
            "trajectory_records_complete": True,
        },
        {
            "minimum_oracle_headroom": "RESEARCH_OWNER_DECISION_REQUIRED",
            "maximum_missing_observation_rate": 0.0,
        },
    )
    assert status == "FAIL"
    assert checks == {"thresholds_frozen": False}


def test_answerer_flip_threshold_is_preserved() -> None:
    thresholds = {"maximum_flip_rate": 0.02}
    passing = {
        "answer_flip_rate": 0.02,
        "label_flip_rate": 0.02,
        "repeats_complete": True,
    }
    assert evaluate_gate("answerer_stability", passing, thresholds)[0] == "PASS"
    assert (
        evaluate_gate(
            "answerer_stability", {**passing, "label_flip_rate": 0.02001}, thresholds
        )[0]
        == "FAIL"
    )


def test_leakage_requires_zero_confirmed_cases() -> None:
    assert (
        evaluate_gate(
            "zero_leakage",
            {"confirmed_leakage": 0, "video_level_isolation": True},
            {},
        )[0]
        == "PASS"
    )
    assert (
        evaluate_gate(
            "zero_leakage",
            {"confirmed_leakage": 1, "video_level_isolation": True},
            {},
        )[0]
        == "FAIL"
    )


def test_dagger_uses_existing_registered_thresholds() -> None:
    thresholds = {
        "minimum_rounds": 2,
        "utility_gain_threshold": 0.005,
        "regret_improvement_ratio": 0.02,
    }
    result = {
        "rounds_complete": 2,
        "utility_gain": 0.005,
        "regret_improvement_ratio": 0.02,
    }
    assert evaluate_gate("dagger", result, thresholds)[0] == "PASS"
