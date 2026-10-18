import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from cbench.dashboard import frontier
from common import FAMILY_COLORS, FAMILY_NAMES, FRONTIER_COLOR, LAYOUT, fmt, gap_text, hover, no_rows, selection

COST_TICKS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]

sel = selection()
metrics, level, all_rows, rows, overall_best = sel.metrics, sel.level, sel.all_rows, sel.rows, sel.best


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


def leaderboard_table(rows: list[dict]) -> None:
    frame = pd.DataFrame([{
        "System": r["label"],
        "Family": FAMILY_NAMES[r["family"]],
        "AUROC": r["auroc"],
        "95% CI": f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]",
        "Gap to best": gap_text(r),
        "Real gap": r["gap_real"],
        "F1": r["f1"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        **({"Hit@1": r["hit1"]} if level == "doc_level" else {"Order gap": r["order_gap"]}),
        "Cost": r["cost"],
    } for r in rows])
    style = frame.style.apply(
        lambda col: ["color: #b91c1c; font-weight: 600" if real else "" for real in frame["Real gap"]],
        subset=["Gap to best"])
    st.dataframe(style, hide_index=True, width="stretch", column_config={
        "AUROC": st.column_config.NumberColumn(format="%.3f"),
        "F1": st.column_config.NumberColumn(format="%.3f"),
        "Precision": st.column_config.NumberColumn(format="%.2f"),
        "Recall": st.column_config.NumberColumn(format="%.2f"),
        "Hit@1": st.column_config.NumberColumn(format="%.2f"),
        "Order gap": st.column_config.NumberColumn(format="%.3f"),
        "Cost": st.column_config.NumberColumn("Cost (est.)", format="$%.3f"),
        "Real gap": st.column_config.CheckboxColumn(help="Gap interval excludes 0"),
    })


def card(col, label: str, value, note: str) -> None:
    col.metric(label, value, note, delta_color="off", delta_arrow="off", border=True)


def leaderboard() -> None:
    st.title("Leaderboard")
    st.caption(f"ContraDoc test split · {metrics.get('n_test_docs', '–')} documents · thresholds tuned for F1 on dev")
    if not rows:
        no_rows()
    best_nli = next((r for r in all_rows if r["family"] == "nli"), None)
    best_decision = next((r for r in all_rows if r["family"] == "decision"), None)
    cols = st.columns(4)
    card(cols[0], "Best AUROC", fmt(overall_best["auroc"]), overall_best["label"])
    if best_nli:
        card(cols[1], "Best NLI", fmt(best_nli["auroc"]), best_nli["label"])
    if best_decision:
        card(cols[2], "Best decision model", fmt(best_decision["auroc"]), best_decision["label"])
    if level == "doc_level" and (yes := metrics.get("always_yes")):
        card(cols[3], "Always-yes F1", fmt(yes["f1"]), "floor")
    else:
        card(cols[3], "Systems", len(all_rows), f"{len(metrics.get('incomplete', []))} partial runs hidden")

    view = st.segmented_control("Chart", ["Cost vs AUROC", "Ranked"], default="Cost vs AUROC",
                                label_visibility="collapsed") or "Cost vs AUROC"
    if view == "Cost vs AUROC":
        if unpriced := [r["label"] for r in rows if r["cost"] is None]:
            st.caption(f"No cost estimate for {len(unpriced)} systems (needs `data/work` and the catalogue); "
                       "they appear in the ranked chart and table only.")
        st.plotly_chart(cost_chart(rows), width="stretch")
        st.caption("Bottom right is best value. Outlined points form the frontier: nothing else is both cheaper "
                   "and better. The dotted line marks the best free model.")
    else:
        st.plotly_chart(ranked_chart(rows), width="stretch")

    leaderboard_table(rows)
    st.caption(f"Gap to best is a paired bootstrap of the AUROC difference to **{overall_best['label']}**, the best "
               "system overall. Red means its interval excludes 0. With many systems, expect a few by chance.")
    if partial := metrics.get("incomplete"):
        with st.expander(f"{len(partial)} partial runs not reported"):
            st.dataframe(pd.DataFrame(partial, columns=["System", "Input", "Scored"]), hide_index=True)



leaderboard()
