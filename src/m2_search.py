from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except (ImportError, RuntimeError, ValueError):
        return " ".join(text.replace("_", " ").split())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c.get("text", "")).split()
                              for c in self.documents]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(self.documents[i].get("text", ""), float(scores[i]),
                         self.documents[i].get("metadata", {}), "bm25")
            for i in indices[:top_k] if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except (ConnectionError, OSError, RuntimeError, ValueError):
            self.client = QdrantClient(":memory:")
        self._encoder = None
        self._vectors = []
        self._documents = []
        self._collection = COLLECTION_NAME

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except (ImportError, OSError, RuntimeError, ValueError):
                self._encoder = False
        return self._encoder

    def _fallback_embedding(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vector_size = 8
        vectors = []
        for text in texts:
            token_scores = {}
            for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
                token_scores[token] = token_scores.get(token, 0) + 1
            keys = sorted(token_scores)
            vector = [0.0] * vector_size
            if keys:
                for idx, key in enumerate(keys[:vector_size]):
                    vector[idx] = float(token_scores[key])
            vectors.append(vector)
        return vectors

    def _embed(self, texts: list[str]) -> list[list[float]]:
        encoder = self._get_encoder()
        if encoder:
            try:
                vectors = encoder.encode(texts, show_progress_bar=False)
                return vectors.tolist() if hasattr(vectors, "tolist") else [list(v) for v in vectors]
            except Exception:
                pass
        return self._fallback_embedding(texts)

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams
        self._collection = collection
        self._documents = list(chunks)
        if not self._documents:
            return
        vectors = self._embed([c.get("text", "") for c in self._documents])
        self._vectors = vectors
        try:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=len(self._vectors[0]), distance=Distance.COSINE),
            )
            points = [
                PointStruct(id=index, vector=vector,
                            payload={**doc.get("metadata", {}), "text": doc.get("text", "")})
                for index, (vector, doc) in enumerate(zip(self._vectors, self._documents))
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception:
            pass

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        query_vector = self._embed([query])[0]
        try:
            response = self.client.query_points(
                collection_name=collection, query=query_vector, limit=top_k
            )
            return [
                SearchResult(pt.payload.get("text", ""), float(pt.score),
                             dict(pt.payload), "dense")
                for pt in response.points
            ]
        except Exception:
            scored = []
            for doc in self._documents:
                text = str(doc.get("text", ""))
                q_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
                d_tokens = set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
                overlap = len(q_tokens & d_tokens)
                score = overlap / max(len(q_tokens) or 1, 1)
                scored.append((score, doc))
            ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
            return [
                SearchResult(doc.get("text", ""), float(score), doc.get("metadata", {}), "dense")
                for score, doc in ranked
            ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0 or top_k <= 0:
        return []
    fused: dict[str, tuple[float, SearchResult]] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            score, original = fused.get(result.text, (0.0, result))
            fused[result.text] = (score + 1.0 / (k + rank + 1), original)
    ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)
    return [SearchResult(result.text, score, result.metadata, "hybrid")
            for score, result in ranked[:top_k]]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
