"""Shared loading, selection and formatting for the explorer pages."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from cbench.dashboard import system_rows
from cbench.paths import RESULTS

CATALOG = RESULTS / "catalog.json"
FAMILY_COLORS = {"decision": "#4f46e5", "nli": "#0d9488"}
FAMILY_NAMES = {"decision": "Decision model", "nli": "NLI (local, free)"}
SERIES_COLORS = ["#4f46e5", "#0d9488", "#f59e0b", "#db2777", "#64748b"]
FRONTIER_COLOR = "#d97706"
LEVELS = {"Documents": "doc_level", "Pairs": "pair_level"}
LAYOUT = dict(template="plotly_white", font=dict(family="Inter, system-ui, sans-serif", size=12),
              margin=dict(l=10, r=10, t=10, b=10), hoverlabel=dict(bgcolor="#111827", font_color="#f9fafb"))


def metrics_path() -> Path:
    return Path(os.environ.get("CBENCH_METRICS", RESULTS / "metrics.json"))


@st.cache_data
def load_metrics(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data
def load_prices() -> dict[str, float]:
    if not CATALOG.exists():
        return {}
    return {m["id"]: m["price_per_token"] for m in json.loads(CATALOG.read_text())["models"]}


@st.cache_data(show_spinner="Estimating token counts…")
def load_tokens() -> dict[str, int] | None:
    from cbench.sweep import TESTS, estimated_tokens

    try:
        return {test: estimated_tokens(test) for test in TESTS}
    except (OSError, SystemExit):
        return None


def fmt(value: float | None, digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def gap_text(row: dict) -> str:
    if row["is_best"]:
        return "best"
    if row["gap"] is None:
        return "–"
    return f"{row['gap']:.3f} [{row['gap_lo']:.3f}, {row['gap_hi']:.3f}]"


def hover(row: dict) -> str:
    cost = "free (local)" if row["cost"] == 0 else ("–" if row["cost"] is None else f"${row['cost']:.3f} (est.)")
    return (f"<b>{row['label']}</b><br>AUROC {fmt(row['auroc'])} [{fmt(row['ci_lo'])}, {fmt(row['ci_hi'])}]"
            f"<br>Gap to best: {gap_text(row)}<br>F1 {fmt(row['f1'])} · cost {cost}")


def matches_mode(row: dict, mode: str) -> bool:
    if mode == "Both" or row["config"] is None:
        return True
    return row["config"].startswith("native:") == (mode == "Native")




@dataclass
class Selection:
    metrics: dict
    level: str
    all_rows: list[dict]
    rows: list[dict]
    best: dict | None


def selection() -> Selection:
    path = metrics_path()
    metrics = load_metrics(str(path), path.stat().st_mtime)
    level = LEVELS[st.session_state.get("test") or "Documents"]
    mode = (st.session_state.get("mode") or "Both") if level == "doc_level" else "Both"
    families = st.session_state.get("families", list(FAMILY_NAMES))
    all_rows = system_rows(metrics, level, load_prices(), load_tokens())
    rows = [r for r in all_rows if r["family"] in families and matches_mode(r, mode)]
    return Selection(metrics, level, all_rows, rows, next((r for r in all_rows if r["is_best"]), None))


def no_rows() -> None:
    st.info("No system has a complete run for this selection yet.")
    st.stop()


