# chatbot_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

app = FastAPI()

# Load FAISS index and chunk mapping at startup
faiss_index = faiss.read_index("embeddings/faiss_index")
with open("embeddings/chunk_mapping.pkl", "rb") as f:
    chunk_mapping = pickle.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_query(request: QueryRequest):
    query_embedding = model.encode([request.question])

    # Search in FAISS index (k=1 means return top 1 nearest chunk)
    k = 1
    distances, indices = faiss_index.search(np.array(query_embedding), k)

    # Retrieve the matching chunk(s)
    matched_chunks = [chunk_mapping[i] for i in indices[0]]

    # Return the top result
    return {
        "question": request.question,
        "answer": matched_chunks[0],
        "distance": float(distances[0][0])
    }

# Run the API:
# uvicorn chatbot_api:app --reload

