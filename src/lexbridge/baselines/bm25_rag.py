from .common import RAGBaseline


class BM25RAG(RAGBaseline):
    name = "bm25"

    def search(self, query):
        return self.index.bm25_plain.search(query, 30)
