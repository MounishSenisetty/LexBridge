"""Cross-encoder rerank (bge-reranker → ms-marco MiniLM) with score passthrough as the CPU fallback."""
from __future__ import annotations

from loguru import logger


class Reranker:
    def __init__(self, cfg=None):
        self.model = None
        self.name = "passthrough"
        models = (cfg or {}).get("models", {})
        if not models.get("offline", True):  # pragma: no cover - needs model download
            for name in (models.get("reranker"), models.get("reranker_fallback")):
                try:
                    from sentence_transformers import CrossEncoder
                    self.model, self.name = CrossEncoder(name, device="cpu"), name
                    break
                except Exception as e:
                    logger.warning(f"reranker {name} unavailable: {type(e).__name__}")

    def rerank(self, query: str, items: list[tuple[int, float]], texts: list[str], top: int = 8):
        if self.model is None or not items:
            return items[:top]
        scores = self.model.predict([(query, texts[i]) for i, _ in items])  # pragma: no cover
        return sorted(((i, float(s)) for (i, _), s in zip(items, scores)), key=lambda x: -x[1])[:top]
