import os
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
from FlagEmbedding import FlagReranker


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_reranker = None
_reranker_lock = Lock()


def rerank_chunks(question: str, candidates: list[dict], final_k: int = 5) -> list[dict]:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not candidates:
        return []

    if final_k <= 0:
        raise ValueError("final_k must be greater than 0.")

    pairs = [[question, candidate["text"]] for candidate in candidates]
    with _reranker_lock:
        scores = get_reranker().compute_score(pairs, normalize=True)

    if not isinstance(scores, list):
        scores = [scores]

    scored_candidates = [{**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    ranked_candidates = sorted(
        scored_candidates,
        key=lambda candidate: candidate["rerank_score"],
        reverse=True,
    )[:final_k]
    return [
        {
            **candidate,
            "retrieval_rank": candidate["source_id"],
            "source_id": rerank_rank,
        }
        for rerank_rank, candidate in enumerate(
            ranked_candidates,
            start=1,
        )
    ]

def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(MODEL_NAME)
    return _reranker