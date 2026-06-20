# RAG System — arXiv Abstracts
End-to-end RAG pipeline for semantic search and Q&A over arXiv paper abstracts.

## Versions

| Version | Pipeline | Key changes |
|---|---|---|
| v1 | Loader → Embedder → FAISS → Retriever → Generator | Baseline. Retriever integrates all components internally. No metadata, no chunking. `bge-small` embedding model |
| v2 | Loader → Embedder → FAISS → Retriever → ContextBuilder → Generator | Modular pipeline. Added metadata (id, title, category). Added chunking (200-word chunks, no overlap). Added ContextBuilder module. `bge-small` embedding model |
| v3 | Loader → Embedder → FAISS → Retriever → Reranker → Generator | Removed chunking (abstracts are self-contained and chunking fragmented them mid-sentence, degrading retrieval and generation quality). Added Reranker (CrossEncoder top-50). Added citation validation. Improved prompt with explicit allowed IDs. Upgraded to `bge-base` embedding model |

## Dataset
[arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) (not included). Place it at `Data/arxiv-metadata-oai-snapshot.json`. Only title and abstract are used.

## Setup
pip install -r requirements.txt

export HF_TOKEN=your_token_here

## Usage
python main.py

## Design decisions (v3)

**No chunking** — abstracts are short (100-250 words) and self-contained. Chunking fragments them mid-sentence, losing semantic context and degrading both retrieval and generation quality. Each paper is indexed as a single unit: title + abstract.

**Reranker** — a CrossEncoder reranks the top-50 FAISS candidates before passing context to the generator, improving precision significantly over dense retrieval alone.

**Citation validation** — the generator output is post-processed to remove any citation IDs not present in the retrieved context, preventing hallucinated references.

## Known limitations
- **Abstracts only** — answers are limited to what abstracts contain. Detailed "how" questions are out of scope as abstracts describe contributions but rarely explain mechanisms in depth.
- **7B model** — Qwen2-7B-Instruct follows formatting instructions inconsistently. Bullet format may vary across queries.
- Both limitations are addressed in the next project, which downloads full PDFs via the arXiv API and uses a stronger generator model.

## Models
- Embedding: `BAAI/bge-base-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generation: `Qwen/Qwen2-7B-Instruct`