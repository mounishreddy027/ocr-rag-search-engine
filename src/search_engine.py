import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

class HybridSearchEngine:
    def __init__(self):
        print("Loading AI Models... (This may take a moment)")
        # 1. Bi-Encoder for Dense Vector Search (Fast)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)

        # 2. Cross-Encoder for Reranking (Accurate)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Storage
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        
        # Sparse Search (BM25) State
        self.bm25 = None
        self.tokenized_corpus = []

    def add_documents(self, chunks: List[str], metadata: Dict[str, Any]):
        if not chunks:
            return

        # A. Update Dense Index (FAISS)
        vectors = self.encoder.encode(chunks)
        self.index.add(np.array(vectors, dtype=np.float32))

        # B. Update Storage & Sparse Data
        for chunk in chunks:
            self.documents.append(chunk)
            self.metadatas.append(metadata)
            self.tokenized_corpus.append(chunk.lower().split())

        # C. Rebuild BM25 Index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.documents:
            return []

        # --- Phase 1: Hybrid Retrieval ---
        
        # 1. Dense Search (Semantic)
        query_vector = self.encoder.encode([query])
        distances, dense_indices = self.index.search(np.array(query_vector, dtype=np.float32), top_k)
        dense_hits = {idx for idx in dense_indices[0] if idx != -1}

        # 2. Sparse Search (Keyword)
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices from sparse scores
        sparse_hits = set(np.argsort(sparse_scores)[::-1][:top_k])

        # 3. Merge Candidates
        candidate_indices = list(dense_hits | sparse_hits)
        candidate_indices = [i for i in candidate_indices if i < len(self.documents)]

        if not candidate_indices:
            return []

        # --- Phase 2: Reranking ---
        
        # Prepare inputs for Cross-Encoder: [[query, doc_text], ...]
        pairs = [[query, self.documents[idx]] for idx in candidate_indices]
        rerank_scores = self.reranker.predict(pairs)

        # Zip results, sort by Reranker score (descending)
        ranked_results = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Format Output
        final_results = []
        for idx, score in ranked_results[:top_k]:
            final_results.append({
                "text": self.documents[idx],
                "score": float(score),
                "metadata": self.metadatas[idx]
            })

        return final_results