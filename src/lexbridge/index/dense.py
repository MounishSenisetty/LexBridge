"""Dense retrieval. Tries sentence-transformers (bge-m3 → MiniLM) unless ``models.offline``;
falls back to a built-in TF-IDF (word + char n-gram) embedder. Vector search: numpy cosine
(faiss / qdrant are optional and only used if installed and configured)."""
from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class TfidfEmbedder:
    name = "builtin-tfidf"

    def __init__(self):
        self.vec = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)),
        ])

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self.vec.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        m = self.vec.transform(texts).toarray().astype(np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(n, 1e-9)


class STEmbedder:  # pragma: no cover - needs model download
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name

    def fit(self, texts):
        return self

    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=16), dtype=np.float32)


def make_embedder(cfg) -> object:
    models = cfg.get("models", {})
    if not models.get("offline", True):
        for name in (models.get("embedder"), models.get("embedder_fallback")):
            try:
                emb = STEmbedder(name)
                logger.info(f"dense embedder: {name}")
                return emb
            except Exception as e:  # pragma: no cover
                logger.warning(f"could not load {name}: {type(e).__name__}")
    logger.info("dense embedder: built-in TF-IDF fallback")
    return TfidfEmbedder()


class DenseIndex:
    def __init__(self, embedder, texts: list[str]):
        self.embedder = embedder.fit(texts) if hasattr(embedder, "fit") else embedder
        self.matrix = self.embedder.encode(texts) if texts else np.zeros((0, 1), np.float32)

    def search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        if not len(self.matrix):
            return []
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
