# main.py — FastAPI entry point for the RAG agent backend
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import agent          # your LangGraph agent
from src.ingestion import ingest_pdf
import uvicorn, os

app = FastAPI(title="RAG Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

class QueryResponse(BaseModel):
    answer: str
    query_type: str
    steps: list
    chunks_used: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = agent.invoke({
            "query":           req.query,
            "query_type":      "",
            "retrieved_chunks": [],
            "answer":          "",
            "steps":           [],
            "chat_history":    req.chat_history
        })
        return QueryResponse(
            answer     = result["answer"],
            query_type = result["query_type"],
            steps      = result["steps"],
            chunks_used= len(result["retrieved_chunks"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)