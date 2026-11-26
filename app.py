"""Streamlit web UI.  Run:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_data_analyst import DataAnalystAgent, Database, build_llm, load_settings
from ai_data_analyst.database import sanitize_identifier, schema_to_text
from ai_data_analyst.llm import LLMError

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_sales.csv"
EXAMPLES = [
    "What are total revenue and profit by region?",
    "Show the monthly revenue trend by category",
    "Top 5 products by profit",
    "Average discount by customer segment",
    "Which category has the highest average order revenue?",
]

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 Autonomous AI Data Analyst")
st.caption("Ask questions about your data in plain English. The agent writes the SQL, runs it, charts it and explains it.")


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload a CSV", type="csv")
    provider = st.selectbox(
        "LLM provider",
        ["auto", "anthropic", "openai", "offline"],
        help="'auto' picks Anthropic or OpenAI if an API key is set, otherwise the offline rule-based demo mode.",
    )


def build_session(source: str, df: pd.DataFrame | None, provider_name: str):
    settings = load_settings(None if provider_name == "auto" else provider_name)
    db = Database(max_rows=settings.max_rows)
    if df is None:
        table = db.load_csv(SAMPLE_CSV)
    else:
        table = db.load_dataframe(df, sanitize_identifier(Path(source).stem))
    return DataAnalystAgent(db, build_llm(settings), settings), table


if upload is not None:
    source, frame = upload.name, pd.read_csv(upload)
else:
    source, frame = "sample_sales.csv", None

key = (source, len(frame) if frame is not None else 0, provider)
if st.session_state.get("session_key") != key:
    try:
        st.session_state.agent, st.session_state.table = build_session(source, frame, provider)
        st.session_state.session_key = key
        st.session_state.history = []
    except LLMError as exc:
        st.error(f"LLM setup error: {exc}")
        st.stop()

agent: DataAnalystAgent = st.session_state.agent
st.sidebar.success(f"Table: `{st.session_state.table}`  |  LLM: `{agent.llm.name}`")
if agent.llm.name == "offline":
    st.sidebar.info("Offline mode handles common questions only. Add an API key to .env for full natural-language support.")
with st.sidebar.expander("Schema"):
    st.code(schema_to_text(agent.db.describe()), language="text")

# ------------------------------------------------------------------ question
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
clicked = None
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, width="stretch"):
        clicked = ex

question = st.chat_input("Ask a question about your data...") or clicked
if question:
    with st.spinner("Analysing..."):
        st.session_state.history.insert(0, agent.ask(question))

for res in st.session_state.history:
    with st.container(border=True):
        st.subheader(res.question)
        if not res.ok:
            st.error(res.error)
            continue
        left, right = st.columns([1, 1])
        with left:
            st.markdown(res.summary)
            with st.expander("SQL"):
                st.code(res.sql, language="sql")
                if len(res.attempts) > 1:
                    st.caption(f"Self-corrected after {len(res.attempts) - 1} failed attempt(s).")
        with right:
            if res.chart_path:
                st.image(str(res.chart_path))
        st.dataframe(res.data, width="stretch", hide_index=True)
        st.download_button("Download CSV", res.data.to_csv(index=False), file_name="result.csv", key=f"dl-{id(res)}")
