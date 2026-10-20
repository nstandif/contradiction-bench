import pandas as pd
import streamlit as st

from cbench.dashboard import explain_rows
from charts import cost_chart, explain_chart, ranked_chart
from common import FAMILY_NAMES, fmt, gap_text, load_prices, load_tokens, no_rows, selection

sel = selection()
metrics, level, all_rows, rows, overall_best = sel.metrics, sel.level, sel.all_rows, sel.rows, sel.best


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


def explain_section() -> None:
    families = st.session_state.get("families", list(FAMILY_NAMES))
    explained = [r for r in explain_rows(metrics, load_prices(), load_tokens()) if r["family"] in families]
    if not explained:
        return
    st.subheader("Naming the pair")
    st.caption(f"{explained[0]['n']} contradictory test documents with a labelled partner sentence. "
               "Find → explain takes the sentence the finders rate highest and the partner the verifiers name for "
               "it; pairwise systems use their top candidate pairs, so candidate recall caps them. "
               "Native runs score sentences only, so they can't name a pair.")
    st.plotly_chart(explain_chart(explained), width="stretch")
    st.dataframe(pd.DataFrame([{
        "System": r["label"], "Family": FAMILY_NAMES[r["family"]], "Pair@1": r["pair1"],
        "95% CI": f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]", "Pair@2": r["pair2"], "Cost": r["cost"],
    } for r in explained]), hide_index=True, width="stretch", column_config={
        "Pair@1": st.column_config.NumberColumn(format="%.3f"),
        "Pair@2": st.column_config.NumberColumn(format="%.3f"),
        "Cost": st.column_config.NumberColumn("Cost (est.)", format="$%.3f"),
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
    if level == "doc_level":
        explain_section()
    if partial := metrics.get("incomplete"):
        with st.expander(f"{len(partial)} partial runs not reported"):
            st.dataframe(pd.DataFrame(partial, columns=["System", "Input", "Scored"]), hide_index=True)



leaderboard()
