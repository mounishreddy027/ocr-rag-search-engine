from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Import our custom modules
from src.ocr_engine import OCRProcessor
from src.text_processor import TextChunker
from src.search_engine import HybridSearchEngine

app = FastAPI(title="OCR RAG System", version="1.1.0")

# Initialize Pipeline Components (Singletons)
ocr = OCRProcessor()
chunker = TextChunker()
engine = HybridSearchEngine()

# --- Structured Output Models ---
class SearchResult(BaseModel):
    text: str
    score: float
    filename: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class BatchIngestResponse(BaseModel):
    processed_count: int
    details: List[str]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# --- API Endpoints ---

@app.post("/ingest/", response_model=BatchIngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Batch Processing: Upload multiple images at once (PNG, JPG, TIFF).
    """
    processed_count = 0
    details = []

    for file in files:
        # 1. Validate File Type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            details.append(f"{file.filename}: Skipped (Invalid Format)")
            continue

        content = await file.read()
        
        # 2. OCR Processing
        raw_text = ocr.extract_text(content)
        if not raw_text.strip():
            details.append(f"{file.filename}: Skipped (No text detected)")
            continue

        # 3. Text Splitting
        chunks = chunker.split_text(raw_text)

        # 4. Indexing (Hybrid)
        engine.add_documents(chunks, metadata={"filename": file.filename})
        
        processed_count += 1
        details.append(f"{file.filename}: Success ({len(chunks)} chunks)")

    return {
        "processed_count": processed_count,
        "details": details
    }

@app.post("/search/", response_model=SearchResponse)
async def search(request: QueryRequest):
    """
    Performs Hybrid Search + Reranking.
    """
    results = engine.search(request.query, request.top_k)
    
    # Map to Structured Output
    structured_results = [
        SearchResult(
            text=r['text'],
            score=r['score'],
            filename=r['metadata']['filename']
        ) for r in results
    ]

    return {
        "query": request.query,
        "results": structured_results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)