"""Plotly figures shared by the explorer pages and app/export_charts.py."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cbench.dashboard import frontier
from common import FAMILY_COLORS, FAMILY_NAMES, FRONTIER_COLOR, LAYOUT, hover

COST_TICKS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]


def cost_chart(rows: list[dict]) -> go.Figure:
    paid = [r for r in rows if r["cost"]]
    free = [r for r in rows if r["cost"] == 0]
    edge = {r["name"] for r in frontier(rows)}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.04)
    for row_index, group in ((1, paid), (2, free)):
        for fam in FAMILY_NAMES:
            points = [r for r in group if r["family"] == fam]
            if not points:
                continue
            fig.add_trace(go.Scatter(
                x=[r["auroc"] for r in points],
                y=[r["cost"] if row_index == 1 else 0 for r in points],
                error_x=dict(type="data", symmetric=False, thickness=1.2, width=0, color=FAMILY_COLORS[fam],
                             array=[r["ci_hi"] - r["auroc"] for r in points],
                             arrayminus=[r["auroc"] - r["ci_lo"] for r in points]),
                mode="markers+text",
                text=[r["label"] if r["name"] in edge or r["is_best"] else "" for r in points],
                textposition="top center", textfont=dict(size=11, color="#334155"),
                marker=dict(color=FAMILY_COLORS[fam], opacity=0.85,
                            size=[11 if r["name"] in edge else 9 for r in points],
                            line=dict(width=[2 if r["name"] in edge else 0 for r in points], color=FRONTIER_COLOR)),
                hovertext=[hover(r) for r in points], hoverinfo="text",
                name=FAMILY_NAMES[fam], legendgroup=fam, showlegend=row_index == 1 or not any(
                    p["family"] == fam for p in paid),
            ), row=row_index, col=1)
    paid_edge = [r for r in frontier(rows) if r["cost"]]
    if paid_edge:
        fig.add_trace(go.Scatter(x=[r["auroc"] for r in paid_edge], y=[r["cost"] for r in paid_edge],
                                 mode="lines", line=dict(color=FRONTIER_COLOR, dash="dash", width=1.5),
                                 name="Best-value frontier", hoverinfo="skip"), row=1, col=1)
    if best := next((r for r in rows if r["is_best"] and r["cost"] is not None), None):
        fig.add_trace(go.Scatter(x=[best["auroc"]], y=[best["cost"] or 0], mode="markers", name="Best AUROC",
                                 marker=dict(symbol="star", size=18, color=FAMILY_COLORS[best["family"]],
                                             line=dict(width=1.5, color="#1e1b4b")),
                                 hovertext=[hover(best)], hoverinfo="text"), row=1 if best["cost"] else 2, col=1)
    if free:
        best_free = max(r["auroc"] for r in free)
        fig.add_vline(x=best_free, line=dict(color="#0d9488", dash="dot", width=1), row="all", col=1)
    fig.update_yaxes(type="log", tickvals=COST_TICKS, ticktext=[f"${t:g}" for t in COST_TICKS],
                     title_text="Estimated cost of run (USD)", row=1, col=1)
    fig.update_yaxes(tickvals=[0], ticktext=["free"], range=[-1, 1], showgrid=False, row=2, col=1)
    fig.update_xaxes(title_text="AUROC (test split) →", row=2, col=1)
    fig.update_layout(**LAYOUT, height=560, legend=dict(orientation="h", y=1.04, x=0))
    return fig


def ranked_chart(rows: list[dict]) -> go.Figure:
    ordered = rows[::-1]
    fig = go.Figure(go.Scatter(
        x=[r["auroc"] for r in ordered], y=[r["label"] for r in ordered], mode="markers",
        marker=dict(color=[FAMILY_COLORS[r["family"]] for r in ordered], size=9,
                    symbol=["star" if r["is_best"] else "circle" for r in ordered]),
        error_x=dict(type="data", symmetric=False, thickness=1.5, width=0, color="#94a3b8",
                     array=[r["ci_hi"] - r["auroc"] for r in ordered],
                     arrayminus=[r["auroc"] - r["ci_lo"] for r in ordered]),
        hovertext=[hover(r) for r in ordered], hoverinfo="text"))
    fig.add_vline(x=0.5, line=dict(color="#cbd5e1", dash="dash"))
    fig.update_xaxes(title_text="AUROC with 95% bootstrap interval")
    fig.update_layout(**LAYOUT, height=max(320, 26 * len(rows) + 60), showlegend=False)
    return fig


def explain_chart(rows: list[dict]) -> go.Figure:
    ordered = rows[::-1]
    labels = [r["label"] for r in ordered]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["pair2"] for r in ordered], y=labels, mode="markers", name="Pair@2",
        marker=dict(symbol="circle-open", size=10, color=[FAMILY_COLORS[r["family"]] for r in ordered]),
        hovertemplate="%{y}<br>Pair@2 %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[r["pair1"] for r in ordered], y=labels, mode="markers", name="Pair@1",
        marker=dict(size=10, color=[FAMILY_COLORS[r["family"]] for r in ordered]),
        error_x=dict(type="data", symmetric=False, thickness=1.5, width=0, color="#94a3b8",
                     array=[r["ci_hi"] - r["pair1"] for r in ordered],
                     arrayminus=[r["pair1"] - r["ci_lo"] for r in ordered]),
        hovertemplate="%{y}<br>Pair@1 %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Share of contradictory test documents whose gold pair is named", range=[0, 1])
    fig.update_layout(**LAYOUT, height=max(260, 26 * len(rows) + 80), legend=dict(orientation="h", y=1.08, x=0))
    return fig
