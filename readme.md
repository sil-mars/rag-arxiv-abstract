# RAG System V1 — arXiv Abstracts

End-to-end RAG pipeline with retrieval and cross-encoder reranking over arXiv abstracts.

---

## Pipeline

Loader → Embedder → FAISS Index → Retriever → Reranker → Generator

## Dataset

[arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (not included). Place it at `Data/arxiv-metadata-oai-snapshot.json`. Only abstracts are used.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here
```

## Usage

```bash
python main.py
```

## Notes

- V1 baseline 
- Embeddings and FAISS index are cached on first run
- Models: BAAI/bge-small-en-v1.5 (embedding), cross-encoder (reranking), Qwen2-7B-Instruct (generation)
