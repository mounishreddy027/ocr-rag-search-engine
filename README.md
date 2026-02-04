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
```bash
git clone [https://github.com/YOUR_USERNAME/ocr-rag-search-engine.git](https://github.com/YOUR_USERNAME/ocr-rag-search-engine.git)
cd ocr-rag-search-engine
