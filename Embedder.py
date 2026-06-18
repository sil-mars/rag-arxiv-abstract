import numpy as np
from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    
    def embed(self, texts, batch_size=32): 
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def save(self, emb, path):
        np.save(path + ".npy", emb)
    
    def load(self, path):
        return np.load(path + ".npy")