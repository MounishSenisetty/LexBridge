"""Baselines: no_rag, bm25, dense, hybrid (all code-unaware)."""


def get_system(name: str, res, llm=None):
    from ..generate.pipeline import ABLATIONS, LexBridge
    from .bm25_rag import BM25RAG
    from .dense_rag import DenseRAG
    from .hybrid_rag import HybridRAG
    from .no_rag import NoRAG
    systems = {"no_rag": NoRAG, "bm25": BM25RAG, "dense": DenseRAG, "hybrid": HybridRAG}
    if name == "lexbridge":
        return LexBridge(res, llm=llm)
    if name in ABLATIONS:
        return LexBridge(res, ABLATIONS[name], llm=llm)
    if name not in systems:
        raise ValueError(f"unknown system {name!r}; choose from lexbridge, {', '.join(systems)}")
    return systems[name](res, llm=llm)
