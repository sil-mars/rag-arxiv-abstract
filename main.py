from Loader import Loader
from Embedder import Embedding
from Indexer import Indexer
from Retriever import Retriever
from Generator import Generator
from Reranker import Reranker

import os
import torch

def main():
    os.environ["HF_TOKEN"] = ""
    path = "/net-vol/Data/arxiv-metadata-oai-snapshot.json"
    artifacts_path = "/net-vol/artifacts/"
    os.makedirs(artifacts_path, exist_ok=True)

    # 1. LOAD
    loader = Loader(path)
    data = loader.read_data()

    # 2. EMBEDDINGS
    embedder = Embedding()

    emb_path = os.path.join(artifacts_path, "embedding.npy")
    
    if os.path.exists(emb_path):
        emb = embedder.load(emb_path)
    else:
        emb = embedder.embed(data)
        embedder.save(emb, emb_path)
    
    # 3. INDEX
    indexer = Indexer()

    index_path = os.path.join(artifacts_path, "index.faiss")
    
    if os.path.exists(index_path):
        indexer.load(index_path)
    else:
        indexer.build_FAISS_flatcos(emb)
        indexer.save(index_path)

    # 4. Reranker
    reranker = Reranker()

    # 5. RETRIEVER + GENERATOR
    retriever = Retriever(embedder, indexer, data)
    gen = Generator()

    # 6. TEST
    test_queries = [
        "What are transformers in NLP?",
        "Graph neural networks applications",
        "Quantum computing challenges"
    ]

    for q in test_queries:
        torch.cuda.empty_cache()
        docs, scores = retriever.retrieve(q, 50)

        reranked = reranker.rerank(q, docs)
        top = reranked[:3]
        
        print("\nTOP AFTER RERANKING:")
        for d in top:
            print("SCORE:", d[1])
            print("TITLE:", d[0]["title"])
            print("---")
        
        context = "\n\n".join(
            [f"[{d[0]['id']}] {d[0]['text']}" for d in top]
        )

        print(context[:3000])

        allowed_ids = [d[0]["id"] for d in top]

        print("\nQUERY:", q)
        print(gen.generate(context, q, allowed_ids))


if __name__ == "__main__":
    main()