import pytest

from cbench.dashboard import explain_rows, frontier, parse_name, run_of, system_rows


def result(auroc, vs_best=None, **extra):
    return {
        "auroc_test": auroc,
        "auroc_ci95_test": [auroc - 0.03, auroc + 0.03],
        "f1_test": 0.7,
        "precision_test": 0.6,
        "recall_test": 0.8,
        "accuracy_test": 0.65,
        "n_test": 624,
        **({"vs_best": vs_best} if vs_best else {}),
        **extra,
    }


METRICS = {
    "pair_level": {
        "typesafe/jev-1.14": result(0.9, order_gap=0.02),
        "nli:nli-deberta-v3-xsmall": result(0.7, {"system": "typesafe/jev-1.14", "delta": 0.2, "ci95": [0.1, 0.3]}),
    },
    "doc_level": {
        "typesafe/jev-1.14 [native:document]": result(0.86),
        "typesafe/jev-1.14 [native:verify]": result(
            0.85, pipeline={"finders": ["typesafe/jev-1.14", "nli:x"], "measured_cost": 0.25}
        ),
        "typesafe/jev-1.14 [cand_k20]": result(0.8, {"system": "x", "delta": 0.06, "ci95": [-0.01, 0.12]}),
        "nli:nli-deberta-v3-xsmall [cand_k20]": result(0.75, {"system": "x", "delta": 0.11, "ci95": [0.05, 0.17]}),
    },
}
PRICES = {"typesafe/jev-1.14": 1e-6}
TOKENS = {"pair_test": 100_000, "native": 1_000_000, "cand_k20": 2_000_000}


def test_parse_name_splits_trailing_config_only():
    assert parse_name("typesafe/jev-1.14 [native:document]") == ("typesafe/jev-1.14", "native:document")
    assert parse_name("nli:nli-deberta-v3-xsmall") == ("nli:nli-deberta-v3-xsmall", None)


def test_run_of_maps_both_native_configs_to_one_run():
    assert run_of(None) == "pair_test"
    assert run_of("native:document") == run_of("native:max-sentence") == "native"
    assert run_of("cand_k20") == "cand_k20"


def test_rows_price_each_configuration_by_its_own_test():
    rows = {r["name"]: r for r in system_rows(METRICS, "doc_level", PRICES, TOKENS)}
    assert rows["typesafe/jev-1.14 [native:document]"]["cost"] == pytest.approx(1.0)
    assert rows["typesafe/jev-1.14 [cand_k20]"]["cost"] == pytest.approx(2.0)
    assert rows["nli:nli-deberta-v3-xsmall [cand_k20]"]["cost"] == 0.0
    pair = {r["name"]: r for r in system_rows(METRICS, "pair_level", PRICES, TOKENS)}
    assert pair["typesafe/jev-1.14"]["cost"] == pytest.approx(0.1)


def test_rows_without_prices_have_unknown_cost_for_decision_models():
    rows = {r["name"]: r for r in system_rows(METRICS, "doc_level", {}, None)}
    assert rows["typesafe/jev-1.14 [cand_k20]"]["cost"] is None
    assert rows["nli:nli-deberta-v3-xsmall [cand_k20]"]["cost"] == 0.0


def test_rows_carry_family_labels_and_gap_significance():
    rows = {r["name"]: r for r in system_rows(METRICS, "doc_level", PRICES, TOKENS)}
    best = rows["typesafe/jev-1.14 [native:document]"]
    assert best["is_best"] and best["family"] == "decision" and best["label"] == "jev-1.14 · native"
    assert rows["typesafe/jev-1.14 [cand_k20]"]["gap_real"] is False
    nli = rows["nli:nli-deberta-v3-xsmall [cand_k20]"]
    assert nli["family"] == "nli" and nli["gap_real"] is True and not nli["is_best"]


def test_frontier_starts_from_the_best_free_model():
    rows = [
        {"name": "free-good", "cost": 0.0, "auroc": 0.80},
        {"name": "free-bad", "cost": 0.0, "auroc": 0.60},
        {"name": "cheap-worse", "cost": 0.1, "auroc": 0.79},
        {"name": "mid-better", "cost": 0.3, "auroc": 0.86},
        {"name": "pricey-worse", "cost": 0.9, "auroc": 0.84},
        {"name": "unpriced", "cost": None, "auroc": 0.99},
    ]
    assert [r["name"] for r in frontier(rows)] == ["free-good", "mid-better"]


def test_verify_pipeline_costs_measured_spend_plus_finder_native_runs():
    rows = {r["name"]: r for r in system_rows(METRICS, "doc_level", PRICES, TOKENS)}
    verify = rows["typesafe/jev-1.14 [native:verify]"]
    assert verify["cost"] == pytest.approx(1.25)
    assert verify["label"] == "jev-1.14 · find → verify"
    assert next(r for r in system_rows(METRICS, "doc_level", {}, None) if r["config"] == "native:verify")["cost"] is None


def test_explain_rows_cost_pipelines_and_pairwise_runs():
    metrics = {"explain": {
        "typesafe/jev-1.14 + nli:x [explain]": {"n": 259, "pair@1": 0.6, "pair@1_ci95": [0.55, 0.65], "pair@2": 0.7,
                                               "pipeline": {"finders": ["typesafe/jev-1.14"], "measured_cost": 0.25}},
        "typesafe/jev-1.14 [cand_k20]": {"n": 259, "pair@1": 0.4, "pair@1_ci95": [0.35, 0.45], "pair@2": 0.5},
    }}
    rows = explain_rows(metrics, PRICES, TOKENS)
    assert [r["label"] for r in rows] == ["jev-1.14 + x · find → explain", "jev-1.14 · cand_k20"]
    assert rows[0]["cost"] == pytest.approx(1.25) and rows[1]["cost"] == pytest.approx(2.0)
