# embedding_faiss.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

def save_faiss_index(embeddings, output_path="embeddings/faiss_index"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, output_path)

if __name__ == "__main__":
    chunks = ["This is a test chunk.", "Another sample chunk."]
    embeddings = generate_embeddings(chunks)
    save_faiss_index(np.array(embeddings))
    print("FAISS index saved.")
