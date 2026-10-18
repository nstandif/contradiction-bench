import streamlit as st

from common import FAMILY_NAMES, LEVELS, metrics_path

st.set_page_config(page_title="contradiction-bench", page_icon=":material/compare_arrows:", layout="wide")


def empty_state() -> None:
    st.title("contradiction-bench")
    st.info(f"No results yet: `{metrics_path()}` doesn't exist.")
    st.markdown("Score some systems, then build the report:")
    st.code("uv run python -m cbench.nli sweep\nuv run python -m cbench.sweep run\nuv run python -m cbench.evaluate",
            language="bash")
    st.stop()


if not metrics_path().exists():
    empty_state()

with st.sidebar:
    st.markdown("### contradiction-bench")
    st.segmented_control("Test", list(LEVELS), default="Documents", key="test")
    if st.session_state.get("test") != "Pairs":
        st.segmented_control("Document mode", ["Pairwise", "Both", "Native"], default="Both", key="mode")
    st.pills("Families", list(FAMILY_NAMES), default=list(FAMILY_NAMES), selection_mode="multi",
             format_func=FAMILY_NAMES.get, key="families")

st.navigation([
    st.Page("leaderboard.py", title="Leaderboard", icon=":material/leaderboard:", default=True),
    st.Page("compare.py", title="Compare", icon=":material/compare:"),
]).run()
