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
    """sentence-transformers model with an in-memory cache keyed by text (index variants share most chunks)."""
    _models: dict = {}

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        if model_name not in STEmbedder._models:
            STEmbedder._models[model_name] = SentenceTransformer(model_name, device="cpu")
        self.model = STEmbedder._models[model_name]
        self.name = model_name
        self.cache: dict[str, np.ndarray] = {}

    def __getstate__(self):  # pickle the name + cache, not the torch model
        return {"name": self.name, "cache": self.cache}

    def __setstate__(self, st):
        self.__init__(st["name"])
        self.cache = st["cache"]

    def fit(self, texts):
        return self

    def load_cache(self, path):
        import pickle
        from pathlib import Path
        if Path(path).exists():
            self.cache.update(pickle.load(open(path, "rb")))

    def save_cache(self, path):
        import pickle
        pickle.dump(self.cache, open(path, "wb"))

    def encode(self, texts):
        todo = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if todo:
            vecs = self.model.encode(todo, normalize_embeddings=True, batch_size=32, show_progress_bar=len(todo) > 500)
            self.cache.update(zip(todo, np.asarray(vecs, dtype=np.float32)))
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)


_EMBEDDERS: dict = {}


def make_embedder(cfg) -> object:
    models = cfg.get("models", {})
    key = (models.get("offline", True), models.get("embedder"))
    if key in _EMBEDDERS and not models.get("offline", True):
        return _EMBEDDERS[key]
    if not models.get("offline", True):
        for name in (models.get("embedder"), models.get("embedder_fallback")):
            try:
                emb = STEmbedder(name)
                logger.info(f"dense embedder: {name}")
                _EMBEDDERS[key] = emb
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
