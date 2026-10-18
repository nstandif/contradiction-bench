"""Flatten results/metrics.json into rows for the Streamlit explorer (app/streamlit_app.py).

Costs are estimates: catalogue price per token times `sweep estimate` tokens for
the run a configuration came from. NLI models run locally and cost nothing.
"""

import re

CONFIG = re.compile(r"^(?P<system>.+) \[(?P<config>[^\]]+)\]$")
CONFIG_LABELS = {"native:document": "native", "native:max-sentence": "native max-sentence"}


def parse_name(name: str) -> tuple[str, str | None]:
    if match := CONFIG.match(name):
        return match["system"], match["config"]
    return name, None


def run_of(config: str | None) -> str:
    if config is None:
        return "pair_test"
    return "native" if config.startswith("native:") else config


def family(system: str) -> str:
    return "nli" if system.startswith("nli:") else "decision"


def label(system: str, config: str | None) -> str:
    short = system.removeprefix("nli:").rsplit("/", 1)[-1]
    return f"{short} · {CONFIG_LABELS.get(config, config)}" if config else short


def cost(system: str, config: str | None, prices: dict[str, float], tokens: dict[str, int] | None) -> float | None:
    if family(system) == "nli":
        return 0.0
    if tokens is None or system not in prices or run_of(config) not in tokens:
        return None
    return prices[system] * tokens[run_of(config)]


def system_rows(metrics: dict, level: str, prices: dict[str, float], tokens: dict[str, int] | None) -> list[dict]:
    results = metrics.get(level, {})
    scored = {n: r for n, r in results.items() if r.get("auroc_test") is not None}
    best = max(scored, key=lambda n: scored[n]["auroc_test"], default=None)
    rows = []
    for name, r in scored.items():
        system, config = parse_name(name)
        ci = r.get("auroc_ci95_test") or (None, None)
        vs = r.get("vs_best") or {}
        gap_ci = vs.get("ci95") or (None, None)
        localization = r.get("localization") or {}
        rows.append({
            "name": name,
            "system": system,
            "config": config,
            "family": family(system),
            "label": label(system, config),
            "auroc": r["auroc_test"],
            "ci_lo": ci[0],
            "ci_hi": ci[1],
            "is_best": name == best,
            "gap": vs.get("delta"),
            "gap_lo": gap_ci[0],
            "gap_hi": gap_ci[1],
            "gap_real": gap_ci[0] is not None and (gap_ci[0] > 0 or gap_ci[1] < 0),
            "f1": r.get("f1_test"),
            "precision": r.get("precision_test"),
            "recall": r.get("recall_test"),
            "accuracy": r.get("accuracy_test"),
            "hit1": localization.get("hit@1"),
            "hit3": localization.get("hit@3"),
            "order_gap": r.get("order_gap"),
            "cost": cost(system, config, prices, tokens),
        })
    return sorted(rows, key=lambda row: -row["auroc"])


def frontier(rows: list[dict]) -> list[dict]:
    """Rows no other row beats on both cost and AUROC, cheapest first."""
    priced = sorted((r for r in rows if r["cost"] is not None), key=lambda r: (r["cost"], -r["auroc"]))
    best, out = float("-inf"), []
    for r in priced:
        if r["auroc"] > best:
            out.append(r)
            best = r["auroc"]
    return out
