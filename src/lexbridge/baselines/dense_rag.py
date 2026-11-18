from .common import RAGBaseline


class DenseRAG(RAGBaseline):
    name = "dense"

    def search(self, query):
        return self.index.dense.search(query, 30)
