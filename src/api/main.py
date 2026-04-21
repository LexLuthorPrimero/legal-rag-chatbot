from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.api.rag_chain import ask

app = FastAPI(title="Legal RAG API")

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list

@app.post("/ask", response_model=Answer)
async def ask_question(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Pregunta vacía")
    answer, sources = ask(q.text)
    return Answer(answer=answer, sources=sources)
