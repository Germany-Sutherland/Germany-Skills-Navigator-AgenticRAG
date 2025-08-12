# Germany Skills Navigator — Agentic RAG Demo (Live Search + Fallback)

This portfolio/demo app showcases a multi-agent retrieval + generation pipeline tailored to German 2025–2030 hiring needs.
It uses a live DuckDuckGo search (when available) to fetch real-time job/skill snippets and falls back to a curated local dataset.

Features:
- Planner → Retriever (live ddg / embed / TF-IDF) → QA → Synthesizer → Verifier → Sentiment → Judge
- Evaluation metrics (BLEU, ROUGE, METEOR, BERTScore when available)
- Last-5 history, Top-K retrieval
- Works on Streamlit Cloud (free) with robust fallbacks

Deploy:
1. Ensure `runtime.txt` (python-3.10) and `requirements.txt` present.
2. Deploy on Streamlit Cloud: main file `app.py`.

Author: Amit Kumar
