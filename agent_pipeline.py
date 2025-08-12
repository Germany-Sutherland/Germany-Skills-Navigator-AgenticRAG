# agent_pipeline.py
# Multi-agent pipeline with live DuckDuckGo search fallback:
# PlannerAgent, RetrieverAgent (live ddg -> sentence-embeds/FAISS optional -> TF-IDF), QAAgent, SynthesizerAgent, VerifierAgent, SentimentAgent, JudgeAgent.

import logging, warnings
from typing import List, Dict
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ---------- helpers ----------
def _try_transformers_pipeline(task: str, model: str):
    try:
        from transformers import pipeline
        return pipeline(task, model=model)
    except Exception:
        return None

# ---------- live search helper (DuckDuckGo) ----------
def _try_ddg_search(query: str, max_results: int = 5):
    """
    Use duckduckgo_search.ddg to fetch top results text snippets.
    Returns list of cleaned text snippets.
    """
    try:
        from duckduckgo_search import ddg
    except Exception:
        return None

    try:
        # query tweaks: focus on jobs in Germany
        q = f"{query} jobs Germany"
        results = ddg(q, region='wt-wt', safesearch='Off', time='y', max_results=max_results)
        snippets = []
        if not results:
            return None
        for r in results:
            # ddg returns dicts with 'title','href','body' sometimes
            text = " ".join([str(r.get(k, "")) for k in ("title", "body", "href") if r.get(k)])
            # very short cleaning
            text = " ".join(text.split())
            if text:
                snippets.append(text)
        return snippets[:max_results]
    except Exception:
        return None

# ---------- TF-IDF retriever (guaranteed) ----------
def _load_tfidf_retriever(docs: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vect = TfidfVectorizer().fit(docs)
    doc_vecs = vect.transform(docs)
    return {"vect": vect, "doc_vecs": doc_vecs, "docs": docs}

# ---------- Planner ----------
class PlannerAgent:
    def __init__(self):
        self.gen = _try_transformers_pipeline("text-generation", "distilgpt2")
    def plan(self, query: str) -> List[str]:
        if self.gen:
            prompt = f"User asks: {query}\nWrite 2 concise steps the agent should take."
            try:
                out = self.gen(prompt, max_length=60, num_return_sequences=1)[0]["generated_text"]
                steps_text = out.replace(prompt, "").strip()
                lines = [l.strip("-. 0123456789") for l in steps_text.splitlines() if l.strip()]
                return lines[:3] if lines else ["Retrieve relevant docs", "Answer from context"]
            except Exception:
                pass
        return ["Retrieve relevant docs", "Answer from context"]

# ---------- Retriever ----------
class RetrieverAgent:
    def __init__(self, documents: List[str]):
        self.local_docs = list(documents)
        self.mode = "local"
        # try embedding-based retrieval first (optional)
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embs = self.embedder.encode(self.local_docs, convert_to_numpy=True)
            # try faiss optionally
            try:
                import faiss
                dim = self.embs.shape[1]
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(self.embs.astype("float32"))
                self.mode = "faiss_local"
            except Exception:
                self.mode = "embed_local"
        except Exception:
            # embedding unavailable -> TF-IDF fallback
            self.tf = _load_tfidf_retriever(self.local_docs)
            self.mode = "tfidf_local"
        # try live search capability
        try:
            # just test import; real search happens in retrieve
            import duckduckgo_search
            self.can_live = True
        except Exception:
            self.can_live = False

    def retrieve(self, query: str, k: int = 3, prefer_live: bool = True) -> List[str]:
        import numpy as np
        # 1) live search if allowed and preferred
        if prefer_live and self.can_live:
            live = _try_ddg_search(query, max_results=k)
            if live:
                return live
        # 2) embedding/FAISS/local TF-IDF fallbacks
        if self.mode == "faiss_local":
            q_emb = self.embedder.encode([query], convert_to_numpy=True)
            D, I = self.index.search(q_emb.astype("float32"), k)
            return [self.local_docs[i] for i in I[0] if i < len(self.local_docs)]
        elif self.mode == "embed_local":
            q_emb = self.embedder.encode([query], convert_to_numpy=True)
            norms = self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-9)
            qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
            sims = (norms @ qn.T).squeeze()
            idxs = np.argsort(-sims)[:k]
            return [self.local_docs[int(i)] for i in idxs]
        else:
            vect = self.tf["vect"]
            doc_vecs = self.tf["doc_vecs"]
            qv = vect.transform([query])
            sims = (doc_vecs @ qv.T).toarray().squeeze()
            idxs = sims.argsort()[::-1][:k]
            return [self.tf["docs"][int(i)] for i in idxs]

# ---------- QA ----------
class QAAgent:
    def __init__(self):
        self.qa = _try_transformers_pipeline("question-answering", "distilbert-base-cased-distilled-squad")
    def answer(self, question: str, context: str):
        if not self.qa:
            first = context.split(".")[0] if context else ""
            return {"answer": first or "No answer available (QA model unavailable).", "score": 0.0}
        try:
            out = self.qa({"question": question, "context": context})
            return {"answer": out.get("answer",""), "score": out.get("score",0.0)}
        except Exception:
            return {"answer":"", "score":0.0, "error":"QA failed"}

# ---------- Synthesizer ----------
class SynthesizerAgent:
    def __init__(self):
        self.gen = _try_transformers_pipeline("text-generation", "distilgpt2")
    def synthesize(self, question: str, pieces: List[str]) -> str:
        context = " ||| ".join([p for p in pieces if p])
        if self.gen:
            prompt = f"Question: {question}\nFacts: {context}\nWrite a concise 1-2 sentence answer:"
            try:
                out = self.gen(prompt, max_length=120, num_return_sequences=1)[0]["generated_text"]
                return out.replace(prompt,"").strip()
            except Exception:
                pass
        return " ".join(pieces[:2]) if pieces else "No synthesized answer."

# ---------- Verifier ----------
class VerifierAgent:
    def verify(self, candidate: str, retrieved: List[str]) -> Dict:
        ok = bool(candidate and any(candidate.strip()[:6].lower() in r.lower() for r in retrieved))
        return {"ok": ok, "note":"presence heuristic"}

# ---------- Sentiment ----------
class SentimentAgent:
    def __init__(self):
        self.pipe = _try_transformers_pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
    def get(self, text: str):
        if not text or not self.pipe:
            return {"label":"UNKNOWN", "score":0.0}
        try:
            out = self.pipe(text[:512])[0]
            return {"label": out.get("label","UNKNOWN"), "score": float(out.get("score",0.0))}
        except Exception:
            return {"label":"UNKNOWN", "score":0.0}

# ---------- Judge (small, safe) ----------
class JudgeAgent:
    def __init__(self):
        self.pipe = _try_transformers_pipeline("text2text-generation", "google/flan-t5-small")
    def evaluate(self, question: str, answer: str):
        if not self.pipe:
            return "Judge model not available in runtime."
        prompt = (
            f"You are an impartial reviewer.\nUser asked: {question}\nAI answered: {answer}\n"
            "Give: Accuracy (1-10), Completeness (1-10), Clarity (1-10), and one-line summary."
        )
        try:
            out = self.pipe(prompt, max_length=200, do_sample=False)[0]["generated_text"]
            return out.strip()
        except Exception:
            return "Judge evaluation failed."

# ---------- AgenticRAG orchestrator ----------
class AgenticRAG:
    def __init__(self, documents: List[str]):
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(documents)
        self.qa = QAAgent()
        self.synth = SynthesizerAgent()
        self.verifier = VerifierAgent()
        self.sentiment = SentimentAgent()
        self.judge = JudgeAgent()

    def run(self, question: str, top_k: int = 2):
        plan = self.planner.plan(question)
        retrieved = self.retriever.retrieve(question, k=top_k, prefer_live=True)
        context = "\n\n".join(retrieved)
        qa_out = self.qa.answer(question, context)
        final = self.synth.synthesize(question, [qa_out.get("answer","")] + retrieved)
        verification = self.verifier.verify(qa_out.get("answer",""), retrieved)
        qsent = self.sentiment.get(question)
        asent = self.sentiment.get(final)
        judge = self.judge.evaluate(question, final)
        return {
            "plan": plan,
            "retrieved": retrieved,
            "qa": qa_out,
            "answer": final,
            "verification": verification,
            "question_sentiment": qsent,
            "answer_sentiment": asent,
            "judge": judge
        }
