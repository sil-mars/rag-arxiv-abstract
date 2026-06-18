import faiss
import numpy as np

class Indexer:

    def __init__(self):
        self.index = None

    def build_FAISS_flatcos(self, embeddings):
        # 1. float32 obligatorio
        embeddings = embeddings.astype(np.float32)

        # 2. cosine similarity
        faiss.normalize_L2(embeddings)

        # 3. crear índice
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)

        # 4. añadir vectores
        self.index.add(embeddings)

        print("Total vectors:", self.index.ntotal)

    def search(self, query_embedding, k=5): # k - cuantos resultados recuperar
        # asegurar formato correcto
        query_embedding = query_embedding.astype(np.float32)
        # normalizar query
        faiss.normalize_L2(query_embedding)

        # busqueda y comparación
        scores, indexes = self.index.search(query_embedding, k)

        return scores, indexes

    def save(self, path):
        faiss.write_index(self.index, path)
    
    def load(self, path):
        self.index = faiss.read_index(path)