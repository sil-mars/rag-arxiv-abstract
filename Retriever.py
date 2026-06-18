import os
import numpy as np

from Loader import Loader
from Embedder import Embedding
from Indexer import Indexer

class Retriever:

    def __init__(self, path, embedding_cache="embedding.npy", index_path="index.faiss"):

        self.loader = Loader(path)
        self.embedder = Embedding()
        self.indexer = Indexer()

        self.embedding_cache = embedding_cache
        self.index_path = index_path

        # 1. Load data
        self.data = self.loader.read_data()

        # 2. Embeddings
        if os.path.exists(self.embedding_cache):
            self.emb = self.embedder.load("embedding")
        else:
            self.emb = self.embedder.embed(self.data, 256)
            self.embedder.save(self.emb, "embedding")

        # 3. FAISS index
        if os.path.exists(self.index_path):
            self.indexer.load(self.index_path)
        else:
            self.indexer.build_FAISS_flatcos(self.emb)
            self.indexer.save(self.index_path)

    def retrieve(self, question, k=5):

        query_emb = self.embedder.embed([question])

        scores, indexes = self.indexer.search(query_emb, k)

        docs = [self.data[i] for i in indexes[0]]

        return docs, scores