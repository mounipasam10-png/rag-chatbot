from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import retrieve
from app.utils import generate_response

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    context = retrieve(query.question)
    response = generate_response(query.question, context)
    return {"response": response}
