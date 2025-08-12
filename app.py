# app.py
import streamlit as st
from textwrap import shorten
from job_data import DOCUMENTS, SKILLS
from evaluation_metrics import compute_all

st.set_page_config(page_title="Germany Skills Navigator — Agentic RAG", layout="wide")
st.title("🇩🇪 Germany Skills Navigator — Agentic RAG Demo (Live Search + Fallback)")

@st.cache_resource(show_spinner=False)
def init_agent():
    from agent_pipeline import AgenticRAG
    return AgenticRAG(DOCUMENTS)

agent = init_agent()

# history (last 5)
if "history" not in st.session_state:
    st.session_state["history"] = []

# Sidebar info
st.sidebar.header("About")
st.sidebar.write("Amit Kumar — Agentic RAG demo for German hiring managers. Live portfolio app.")
st.sidebar.header("Top Skills (sample)")
for s in SKILLS[:20]:
    st.sidebar.write(f"- {s}")

# Controls and input
col_main, col_ctrl = st.columns([3,1])
with col_ctrl:
    top_k = st.selectbox("Top-K retrieve (k)", [1,2,3], index=1)
    show_plan = st.checkbox("Show plan", True)
    show_retrieved = st.checkbox("Show retrieved", True)
    show_qa = st.checkbox("Show QA output", False)
    show_verification = st.checkbox("Show verification", True)
    show_sentiment = st.checkbox("Show sentiment", True)
    show_judge = st.checkbox("Show judge review", True)
    show_metrics = st.checkbox("Show evaluation metrics", True)
    prefer_live = st.checkbox("Prefer live web results (DuckDuckGo)", value=True)

with col_main:
    question = st.text_input("Ask a hiring/recruiting question (e.g., 'Which skills for EV manufacturing in Munich?'):")
    if st.button("Run Agent") and question.strip():
        with st.spinner("Agent is planning and executing..."):
            # pass prefer_live into retrieval via agent.retriever; we call run() as usual
            out = agent.run(question, top_k)
        # history
        st.session_state["history"].append(question)
        st.session_state["history"] = st.session_state["history"][-5:]

        # plan
        if show_plan:
            st.subheader("Agent Plan")
            for i, s in enumerate(out["plan"], 1):
                st.write(f"{i}. {s}")

        # final answer
        st.subheader("Final Answer")
        st.success(shorten(out["answer"], width=800))

        # sentiments
        if show_sentiment:
            qsent = out.get("question_sentiment", {})
            asent = out.get("answer_sentiment", {})
            st.markdown("**Sentiment**")
            st.write(f"Question: **{qsent.get('label','UNKNOWN')}** (score {qsent.get('score',0.0):.2f})")
            st.write(f"Answer: **{asent.get('label','UNKNOWN')}** (score {asent.get('score',0.0):.2f})")

        # evaluation metrics
        if show_metrics:
            try:
                metrics = compute_all(question, out["answer"])
                st.markdown("**Evaluation Metrics**")
                for k,v in metrics.items():
                    st.write(f"- **{k}**: {v}")
            except Exception as e:
                st.write("Metrics unavailable:", e)

        # QA output
        if show_qa:
            st.markdown("**QA output**")
            st.json(out.get("qa", {}))

        # verification
        if show_verification:
            st.markdown("**Verification**")
            st.json(out.get("verification", {}))

        # retrieved
        if show_retrieved:
            st.markdown("**Retrieved Passages (live when available, otherwise local)**")
            for i,p in enumerate(out.get("retrieved", []),1):
                st.write(f"Doc {i}: {shorten(p, 400)}")

        # judge
        if show_judge:
            st.markdown("**Judge Review**")
            st.write(out.get("judge", "Judge not available in runtime."))

# history display
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("Last 5 questions")
    for q in reversed(st.session_state["history"]):
        st.write(f"- {q}")

st.markdown("---")
st.caption("Demo built with open-source models; prefers live web results (DuckDuckGo) when available, else falls back to local data.")
