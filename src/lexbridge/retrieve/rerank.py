"""Cross-encoder rerank (bge-reranker → ms-marco MiniLM) with score passthrough as the CPU fallback."""
from __future__ import annotations

from loguru import logger


_CACHE: dict[tuple[str, str], float] = {}
_MODELS: dict = {}


class Reranker:
    def __init__(self, cfg=None):
        self.model = None
        self.name = "passthrough"
        models = (cfg or {}).get("models", {})
        if not models.get("offline", True):  # pragma: no cover - needs model download
            for name in (models.get("reranker"), models.get("reranker_fallback")):
                try:
                    from sentence_transformers import CrossEncoder
                    if name not in _MODELS:
                        _MODELS[name] = CrossEncoder(name, device="cpu", max_length=512)
                    self.model, self.name = _MODELS[name], name
                    break
                except Exception as e:
                    logger.warning(f"reranker {name} unavailable: {type(e).__name__}")

        self.cache = _CACHE
        self.candidates = int(((cfg or {}).get("retrieval", {}) or {}).get("rerank_candidates", 50))

    def rerank(self, query: str, items: list[tuple[int, float]], texts: list[str], top: int = 8):
        if self.model is None or not items:
            return items[:top]
        items = items[: self.candidates]  # pragma: no cover - needs a model
        todo = [(query, texts[i]) for i, _ in items if (query, texts[i]) not in self.cache]
        if todo:
            for pair, s in zip(todo, self.model.predict(todo, batch_size=16)):
                self.cache[pair] = float(s)
        return sorted(((i, self.cache[(query, texts[i])]) for i, _ in items), key=lambda x: -x[1])[:top]
