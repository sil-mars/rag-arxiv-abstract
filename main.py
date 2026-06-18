import os
from Retriever import Retriever
from Generator import Generator

os.environ["HF_TOKEN"] = ""

DATA_PATH = "Data/arxiv-metadata-oai-snapshot.json"

test_queries = [
    "What are the main approaches to transformer models in NLP?",
    "What problems are graph neural networks used to solve?",
    "What are the open challenges in quantum computing?",
]


def main():
    retriever = Retriever(DATA_PATH)
    gen = Generator()

    for q in test_queries:
        docs, scores = retriever.retrieve(q)

        context = "\n\n".join(docs)

        print("\nQUERY:", q)
        print(gen.generate(context, q))


if __name__ == "__main__":
    main()
