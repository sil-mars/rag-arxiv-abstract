# RAG System V2 — arXiv Abstracts

End-to-end RAG pipeline for semantic search and Q&A over arXiv paper abstracts.

---

## Pipeline

Loader → Embedder → FAISS Index → Retriever → ContextBuilder → Generator

## Dataset

[arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (not included). Place it at `Data/arxiv-metadata-oai-snapshot.json`. Only title and abstract are used.

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

- V2: adds chunking (200-word chunks) and a dedicated ContextBuilder module
- Embeddings and FAISS index are cached on first run
- Models: BAAI/bge-small-en-v1.5 (embedding), Qwen2-7B-Instruct (generation)