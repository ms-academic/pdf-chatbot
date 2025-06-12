# chatbot_cli.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load FAISS index and chunk mapping
faiss_index = faiss.read_index("embeddings/faiss_index")
with open("embeddings/chunk_mapping.pkl", "rb") as f:
    chunk_mapping = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_answer(question):
    query_embedding = model.encode([question])
    k = 1  # number of nearest matches
    distances, indices = faiss_index.search(np.array(query_embedding), k)
    matched_chunks = [chunk_mapping[i] for i in indices[0]]
    return matched_chunks[0], distances[0][0]

if __name__ == "__main__":
    print("🔍 PDF Chatbot is ready! Type 'exit' to quit.\n")
    
    while True:
        question = input("❓ You: ")
        if question.lower() == 'exit':
            print("👋 Exiting chatbot. Goodbye!")
            break

        answer, distance = get_answer(question)
        print(f"\n🤖 Chatbot (score: {distance:.2f}): {answer}\n")
