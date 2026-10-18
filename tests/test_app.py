import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="install the app group: uv sync --group app")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def doc_result(auroc, gap=None):
    return {
        "auroc_test": auroc,
        "auroc_ci95_test": [auroc - 0.03, auroc + 0.03],
        "f1_test": 0.7, "precision_test": 0.6, "recall_test": 0.8, "accuracy_test": 0.65, "n_test": 624,
        "n_docs_scored": 891,
        "auroc_by_length_tertile": {"short": auroc, "medium": auroc - 0.05, "long": auroc - 0.1},
        "recall_by": {
            "contra_type": {"Factual": {"n": 40, "recall": 0.8}, "Numeric": {"n": 20, "recall": 0.6}},
            "scope": {"local": {"n": 50, "recall": 0.7}},
            "doc_type": {"news": {"n": 60, "recall": 0.7}},
        },
        "localization": {"n": 300, "hit@1": 0.5, "hit@3": 0.7},
        **({"vs_best": {"system": "best", "delta": gap, "ci95": [gap - 0.02, gap + 0.02]}} if gap else {}),
    }


@pytest.fixture
def metrics_file(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "n_test_docs": 624,
        "pair_level": {
            "typesafe/jev-1.14": {**doc_result(0.9), "order_gap": 0.02,
                                  "auroc_by_type_all_splits": {"Factual": {"n_pos": 40, "auroc": 0.9}}},
        },
        "doc_level": {
            "typesafe/jev-1.14 [native:document]": doc_result(0.86),
            "nli:nli-deberta-v3-xsmall [cand_k20]": doc_result(0.75, 0.11),
        },
        "always_yes": {"f1": 0.669, "accuracy": 0.5, "precision": 0.5, "recall": 1.0},
        "incomplete": [["nli:roberta-large-mnli", "cand_k20", "100/17815"]],
    }))
    monkeypatch.setenv("CBENCH_METRICS", str(path))
    return path


def test_leaderboard_renders_without_errors(metrics_file):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Leaderboard"
    assert [m.value for m in app.metric][:1] == ["0.860"]


@pytest.mark.parametrize("test", ["Documents", "Pairs"])
def test_compare_renders_for_both_tests(metrics_file, test):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.segmented_control(key="test").set_value(test).run()
    app.switch_page("compare.py").run()
    assert not app.exception
    assert app.title[0].value == "Compare"
    assert app.multiselect[0].value


def test_missing_metrics_shows_how_to_build_them(tmp_path, monkeypatch):
    monkeypatch.setenv("CBENCH_METRICS", str(tmp_path / "missing.json"))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert "No results yet" in app.info[0].value
