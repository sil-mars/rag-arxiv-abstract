from Loader import Loader
from Embedder import Embedding
from Indexer import Indexer
from Retriever import Retriever
from Generator import Generator
from ContextB import ContextB

import os
import torch

def main():

	os.environ["HF_TOKEN"] = ""
    path = "Data/arxiv-metadata-oai-snapshot.json"

    # 1. LOAD
    loader = Loader(path)
    data = loader.read_data()

    # 2. EMBEDDINGS
    embedder = Embedding()

    if os.path.exists("embedding.npy"):
        emb = embedder.load("embedding")
    else:
        emb = embedder.embed(data)
        embedder.save(emb, "embedding.npy")

    # 3. INDEX
    indexer = Indexer()

    if os.path.exists("index.faiss"):
        indexer.load("index.faiss")
    else:
        indexer.build_FAISS_flatcos(emb)
        indexer.save("index.faiss")

    # 4. RETRIEVER + GENERATOR
    retriever = Retriever(embedder, indexer, data)
    gen = Generator()
    contextb = ContextB()

    # 5. TEST
    test_queries = [
        "What are transformers in NLP?",
        "Graph neural networks applications",
        "Quantum computing challenges"
    ]

    for q in test_queries:
        torch.cuda.empty_cache()
        docs, scores = retriever.retrieve(q, 3)

        context = contextb.build(docs)

        print("\nQUERY:", q)
        print(gen.generate(context, q))


if __name__ == "__main__":
    main()