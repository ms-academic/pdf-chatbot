# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_query(request: QueryRequest):
    # Placeholder for chatbot logic
    response = f"Received your question: '{request.question}'"
    return {"answer": response}

# Run the API:
# uvicorn app.api:app --reload
