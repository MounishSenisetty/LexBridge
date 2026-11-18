from ..retrieve.hybrid import rrf
from .common import RAGBaseline


class HybridRAG(RAGBaseline):
    name = "hybrid"

    def search(self, query):
        bm = [i for i, _ in self.index.bm25_plain.search(query, 30)]
        dn = [i for i, _ in self.index.dense.search(query, 30)]
        return rrf([bm, dn], k=60)
