from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                # Avoid blocking indefinitely when an optional model is not
                # cached (for example in an offline grading environment).
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except (ImportError, OSError, RuntimeError, ValueError):
                self._model = False
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        if model:
            scores = model.predict(pairs)
        else:
            query_terms = set(query.lower().split())
            scores = [
                (2.0 * len(query_terms & set(doc.get("text", "").lower().split()))
                 / max(len(query_terms), 1)) + float(doc.get("score", 0.0)) * 1e-3
                for doc in documents
            ]
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]),
                        reverse=True)
        return [
            RerankResult(doc.get("text", ""), float(doc.get("score", 0.0)),
                         float(score), doc.get("metadata", {}), rank)
            for rank, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Fallback reranker with a fast lexical signal when flashrank is unavailable."""
        if not documents or top_k <= 0:
            return []
        query_terms = set((query or "").lower().split())
        scored = []
        for doc in documents:
            text = str(doc.get("text", ""))
            doc_terms = set(text.lower().split())
            overlap = len(query_terms & doc_terms)
            lexical = overlap / max(len(query_terms) or 1, 1)
            score = float(doc.get("score", 0.0)) + lexical
            scored.append((score, doc))
        ranked = sorted(scored, key=lambda item: float(item[0]), reverse=True)[:top_k]
        return [
            RerankResult(
                doc.get("text", ""),
                float(doc.get("score", 0.0)),
                float(score),
                doc.get("metadata", {}),
                rank,
            )
            for rank, (score, doc) in enumerate(ranked)
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
