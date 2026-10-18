import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import LAYOUT, SERIES_COLORS, gap_text, no_rows, selection

sel = selection()
metrics, level, rows, overall_best = sel.metrics, sel.level, sel.rows, sel.best


def grouped_bars(series: dict[str, dict[str, float]], value_title: str) -> go.Figure:
    fig = go.Figure()
    for color, (name, values) in zip(SERIES_COLORS, series.items()):
        fig.add_trace(go.Bar(y=list(values), x=list(values.values()), name=name, orientation="h",
                             marker_color=color, hovertemplate="%{y}: %{x:.2f}<extra>" + name + "</extra>"))
    fig.update_layout(**LAYOUT, barmode="group", height=110 + 24 * len(series) * max(map(len, series.values())),
                      legend=dict(orientation="h", y=-0.15), yaxis=dict(autorange="reversed"))
    fig.update_xaxes(range=[0, 1], title_text=value_title)
    return fig


def compare() -> None:
    st.title("Compare")
    if not rows:
        no_rows()
    by_label = {r["label"]: r for r in rows}
    chosen = st.multiselect("Systems", list(by_label), default=list(by_label)[:3], max_selections=len(SERIES_COLORS))
    if not chosen:
        st.info("Pick at least one system.")
        return
    results = {name: metrics[level][by_label[name]["name"]] for name in chosen}

    if level == "pair_level":
        st.subheader("AUROC by contradiction type")
        st.caption("Computed over all splits: each type's gold pairs against every clean-document pair.")
        series = {n: {t: v["auroc"] for t, v in r["auroc_by_type_all_splits"].items() if v["auroc"] is not None}
                  for n, r in results.items()}
        st.plotly_chart(grouped_bars(series, "AUROC"), width="stretch")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Recall by contradiction type")
            st.caption("Test positives at the dev-tuned threshold; a document counts once per type.")
            series = {n: {t: v["recall"] for t, v in r["recall_by"]["contra_type"].items()} for n, r in results.items()}
            st.plotly_chart(grouped_bars(series, "Recall"), width="stretch")
        with right:
            st.subheader("AUROC by document length")
            st.caption("Pairwise systems take the max over more pairs in longer documents, so their scores drift.")
            fig = go.Figure()
            for color, (name, r) in zip(SERIES_COLORS, results.items()):
                tertiles = r["auroc_by_length_tertile"]
                fig.add_trace(go.Scatter(x=list(tertiles), y=list(tertiles.values()), name=name, mode="lines+markers",
                                         line=dict(color=color, width=2.5), marker=dict(size=8)))
            fig.update_layout(**LAYOUT, height=320, legend=dict(orientation="h", y=-0.2))
            fig.update_yaxes(title_text="AUROC")
            st.plotly_chart(fig, width="stretch")
            st.subheader("Recall by scope")
            series = {n: {s: v["recall"] for s, v in r["recall_by"]["scope"].items()} for n, r in results.items()}
            st.plotly_chart(grouped_bars(series, "Recall"), width="stretch")

    st.subheader("Gap to best")
    st.dataframe(pd.DataFrame([{
        "System": name, "AUROC": by_label[name]["auroc"], "Gap to best": gap_text(by_label[name]),
        "Real gap": by_label[name]["gap_real"], "Cost (est.)": by_label[name]["cost"],
    } for name in chosen]), hide_index=True, width="stretch", column_config={
        "AUROC": st.column_config.NumberColumn(format="%.3f"),
        "Cost (est.)": st.column_config.NumberColumn(format="$%.3f"),
    })
    st.caption(f"Each gap is to **{overall_best['label']}**. The report stores paired gaps only against the best "
               "system, so two other systems can't be compared head to head here.")



compare()
