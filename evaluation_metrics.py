# evaluation_metrics.py
# Safe, fallback-friendly evaluation metrics for generated answer vs input question.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Optional heavy metrics -- attempt to import but allow graceful fallback
try:
    import evaluate
except Exception:
    evaluate = None

try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None

def _cosine_tfidf(a: str, b: str) -> float:
    try:
        vect = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(vect[0:1], vect[1:2])[0][0])
    except Exception:
        return 0.0

def _bleu(a: str, b: str) -> float:
    try:
        smoothie = SmoothingFunction().method4
        return float(sentence_bleu([a.split()], b.split(), smoothing_function=smoothie))
    except Exception:
        return 0.0

def _rouge_scores(a: str, b: str):
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
        sc = scorer.score(a, b)
        return {"rouge1": sc["rouge1"].fmeasure, "rougeL": sc["rougeL"].fmeasure}
    except Exception:
        return {"rouge1": 0.0, "rougeL": 0.0}

def _meteor(a: str, b: str) -> float:
    try:
        if evaluate:
            m = evaluate.load("meteor")
            return float(m.compute(predictions=[b], references=[a])["meteor"])
    except Exception:
        return 0.0

def _bert(a: str, b: str):
    try:
        if bert_score:
            P, R, F1 = bert_score([b], [a], lang="en", rescale_with_baseline=True, verbose=False)
            return {"P": float(P.mean()), "R": float(R.mean()), "F1": float(F1.mean())}
    except Exception:
        return {"P": 0.0, "R": 0.0, "F1": 0.0}

def compute_all(reference: str, prediction: str):
    metrics = {}
    metrics["Cosine (TF-IDF)"] = round(_cosine_tfidf(reference, prediction), 4)
    metrics["BLEU"] = round(_bleu(reference, prediction), 4)
    rouge = _rouge_scores(reference, prediction)
    metrics["ROUGE-1"] = round(rouge["rouge1"], 4)
    metrics["ROUGE-L"] = round(rouge["rougeL"], 4)
    metrics["METEOR"] = round(_meteor(reference, prediction), 4)
    bert = _bert(reference, prediction)
    metrics["BERTScore Precision"] = round(bert["P"], 4)
    metrics["BERTScore Recall"] = round(bert["R"], 4)
    metrics["BERTScore F1"] = round(bert["F1"], 4)
    return metrics
