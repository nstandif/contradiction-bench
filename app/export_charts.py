"""Render the explorer's charts to docs/img/ for the README.

uv run --group app python app/export_charts.py
"""

import json

from cbench.dashboard import explain_rows, system_rows
from cbench.paths import ROOT
from charts import cost_chart, explain_chart, ranked_chart
from common import CATALOG, metrics_path

OUT = ROOT / "docs" / "img"
WIDTH = 1100


def main() -> None:
    from cbench.sweep import TESTS, estimated_tokens

    metrics = json.loads(metrics_path().read_text())
    prices = {m["id"]: m["price_per_token"] for m in json.loads(CATALOG.read_text())["models"]}
    tokens = {test: estimated_tokens(test) for test in TESTS}
    docs = system_rows(metrics, "doc_level", prices, tokens)
    pairs = system_rows(metrics, "pair_level", prices, tokens)
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "doc_cost_vs_auroc": cost_chart(docs),
        "doc_ranked": ranked_chart(docs),
        "pair_ranked": ranked_chart(pairs),
        "naming_the_pair": explain_chart(explain_rows(metrics, prices, tokens)),
    }
    for name, fig in figures.items():
        fig.update_layout(margin=dict(l=10, r=20, t=30, b=10))
        fig.write_image(OUT / f"{name}.png", width=WIDTH, height=fig.layout.height, scale=2)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
