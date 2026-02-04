# 📄 AI-Powered OCR Search Engine (RAG)

A production-ready **Retrieval-Augmented Generation (RAG)** system that ingests scanned documents (images), extracts text via OCR, and performs semantic search using a **Hybrid Pipeline (Dense + Sparse Retrieval)** with **Cross-Encoder Reranking**.

Built with **FastAPI**, **Docker**, **PyTorch**, and **LangChain**.

---

## 🚀 Key Features
- **Multi-Modal Ingestion:** Supports batch uploading of images (PNG, JPG, TIFF).
- **OCR Engine:** Tesseract 5.0 optimized for noisy documents (faxes, forms).
- **Hybrid Search:** Combines **FAISS** (Semantic Vector Search) and **BM25** (Keyword Search) for maximum recall.
- **Precision Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score top candidates.
- **Recursive Text Splitting:** Preserves semantic context across document chunks.
- **Dockerized:** Fully containerized environment for consistent deployment.

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Framework:** FastAPI
- **ML/AI:** SentenceTransformers, FAISS, RankBM25, PyTorch (CPU-optimized)
- **Infrastructure:** Docker, Uvicorn

## ⚡ How to Run

### Prerequisite
Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed.

### 1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/ocr-rag-search-engine.git](https://github.com/YOUR_USERNAME/ocr-rag-search-engine.git)
cd ocr-rag-search-engine

###2. Build & Run (Docker)
Bash
docker build -t ocr-rag-pro .
docker run -p 8000:8000 ocr-rag-pro

###3. Use the API
Open your browser to: http://localhost:8000/docs

Ingest: Use POST /ingest/ to upload document images.

Search: Use POST /search/ to query the indexed documents.

🧪 Tested Scenarios
This system has been validated against the FUNSD Dataset (Form Understanding in Noisy Scanned Documents):

Complex Tables: Accurately retrieves row-specific data.

Handwriting: Handles mixed printed/handwritten forms.

Noise: Robust against scan artifacts and fax headers.

📂 Project Structure
├── src/
│   ├── ocr_engine.py       # Tesseract Wrapper
│   ├── text_processor.py   # Recursive Chunking
│   ├── search_engine.py    # Hybrid Search & Reranking Logic
├── app.py                  # FastAPI Entrypoint
├── Dockerfile              # Production Image Config
└── requirements.txt        # Dependencies

### **Step 6: Update the README on GitHub**
After creating the `README.md` file locally:
1.  Run `git add README.md`
2.  Run `git commit -m "Add documentation"`
3.  Run `git push`

Now your GitHub repo will look complete and professional!
