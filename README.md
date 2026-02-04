# AI-Powered OCR Search Engine (RAG)

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-v24-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

A production-ready **Retrieval-Augmented Generation (RAG)** system designed to ingest scanned documents, extract text via OCR, and perform semantic search. It utilizes a **Hybrid Pipeline (Dense + Sparse Retrieval)** and **Cross-Encoder Reranking** to ensure high-precision results even with noisy, real-world data (faxes, invoices, forms).

---

## Key Features

* **Multi-Modal Ingestion:** Supports batch uploading of images (`PNG`, `JPG`, `TIFF`, `BMP`).
* **Robust OCR Engine:** Powered by **Tesseract 5.0**, optimized for extracting text from noisy documents like fax headers and mixed-media forms.
* **Gold Standard Retrieval:**
    * **Hybrid Search:** Combines **FAISS** (Dense Vector Search) for semantic meaning and **BM25** (Sparse Keyword Search) for exact keyword matching.
    * **Precision Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score top candidates, pushing the most relevant context to the top.
* **Recursive Text Splitting:** Preserves semantic context across paragraph and sentence boundaries using LangChain splitters.
* **Dockerized:** Fully containerized environment for consistent, "write once, run anywhere" deployment.

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10 |
| **Framework** | FastAPI (Async/Await) |
| **AI / ML** | SentenceTransformers, FAISS, RankBM25, PyTorch (CPU-optimized) |
| **OCR** | Tesseract OCR (via `pytesseract`) |
| **Infrastructure** | Docker, Uvicorn |

---

## How to Run

### Prerequisite
Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/ocr-rag-search-engine.git](https://github.com/YOUR_USERNAME/ocr-rag-search-engine.git)
cd ocr-rag-search-engine
```

### 2. Build & Run (Docker)
This command builds the image and installs all dependencies (including Tesseract and PyTorch) in an isolated container.
```bash
# Build the image
docker build -t ocr-rag-pro .

# Run the container on port 8000
docker run -p 8000:8000 ocr-rag-pro
```

### 3. Use the API
Open your browser to the interactive Swagger UI:  http://localhost:8000/docs

POST /ingest/: Upload document images (single or batch).

POST /search/: Ask natural language questions about your uploaded documents.

## Validated Scenarios
This system has been validated against the FUNSD Dataset (Form Understanding in Noisy Scanned Documents):

Scenario,Result
Complex Tables,"Accurately retrieves row-specific data (e.g., specific line items in an invoice)."
Handwriting,Handles mixed printed/handwritten forms effectively.
Noise Resilience,"Robust against scan artifacts, fax headers, and poor contrast."

## Project Structure

ocr_rag_search_engine/
├── src/
│   ├── ocr_engine.py       # Tesseract Wrapper
│   ├── text_processor.py   # Recursive Chunking Logic
│   ├── search_engine.py    # Hybrid Search (FAISS + BM25) & Reranking
├── app.py                  # FastAPI Application Entrypoint
├── Dockerfile              # Production Image Config
└── requirements.txt        # Dependencies
