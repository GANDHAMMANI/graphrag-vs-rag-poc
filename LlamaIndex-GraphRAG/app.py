"""
app.py — Simple Streamlit demo UI for the LlamaIndex GraphRAG lane.

Run:
    venv\\Scripts\\streamlit run app.py

First load builds all communities (Leiden clustering + LLM summaries) once
via st.cache_resource, then every question in the session reuses that same
retriever — matches how evaluate.py uses it (one retriever, many queries).
"""

import time

import streamlit as st

from core.retrieve import LlamaIndexGraphRetriever

st.set_page_config(page_title="LlamaIndex GraphRAG Demo", page_icon="🕸️", layout="wide")


@st.cache_resource(show_spinner="Building graph communities (first load only) …")
def get_retriever():
    return LlamaIndexGraphRetriever()


st.title("🕸️ LlamaIndex GraphRAG — Demo")
st.caption("Community-based GraphRAG retrieval over the ingested Neo4j graph.")

retriever = get_retriever()

question = st.text_input(
    "Ask a question",
    placeholder="e.g. Which departments submitted NAAC accreditation documents?",
)
ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving and generating answer …"):
        t0 = time.perf_counter()
        result = retriever.retrieve_with_citations(question)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"{elapsed_ms} ms")

    with st.expander(f"Community summaries used ({len(result['contexts'])})", expanded=False):
        if not result["contexts"]:
            st.write("No community summaries were used for this answer.")
        for i, ctx in enumerate(result["contexts"], 1):
            st.markdown(f"**Community {i}**")
            st.write(ctx)
            st.divider()

elif ask:
    st.warning("Enter a question first.")