from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from src.agent import agent          # your LangGraph agent
from src.ingestion import ingest_pdf
from src.retrieval import initialize_bm25_index # CRUCIAL: ensures indexing refreshes on upload
import uvicorn, os, shutil

app = FastAPI(title="RAG Agent API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

class QueryResponse(BaseModel):
    answer: str
    query_type: str
    steps: list
    chunks_used: int

class IngestionResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int

@app.get("/health")
def health():
    return {"status": "ok"}

# ── NEW: EXPOSED DOCUMENT INGESTION ROUTE ───────────────────────────
@app.post("/ingest", response_model=IngestionResponse)
def ingest_document(file: UploadFile = File(...)):
    # 1. Guard check: enforce PDF uploads strictly
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a valid PDF document.")
    
    # Ensure temporary data sandbox directory exists inside container
    temp_dir = "/app/data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # 2. Stream the incoming upload file directly onto the local container disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Process file through your core ingestion script
        chunk_count = ingest_pdf(temp_file_path)
        
        # 4. Re-calculate your vocabulary statistics so BM25 keyword matching sees the new text
        initialize_bm25_index()
        
        return IngestionResponse(
            status="Success",
            filename=file.filename,
            chunks_created=chunk_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion processing failure: {str(e)}")
        
    finally:
        # 5. Guard Layer: Always delete the scratch file from disk, even if the processing steps throw an error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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