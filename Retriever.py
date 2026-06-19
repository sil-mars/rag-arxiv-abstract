import os
import numpy as np

class Retriever:

    def __init__(self, embedder, indexer, data):
        self.embedder = embedder
        self.indexer = indexer
        self.data = data

    def retrieve(self, question, k=5):

        query_emb = self.embedder.embed([question])

        scores, indexes = self.indexer.search(query_emb, k)

        docs = [self.data[i] for i in indexes[0]]

        return docs, scores