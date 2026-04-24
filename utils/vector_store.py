"""
vector_store.py
---------------
Builds an in-memory FAISS index from text chunks using the
sentence-transformers 'all-MiniLM-L6-v2' model for embeddings.

If sentence-transformers cannot be imported in the environment,
it falls back to a lightweight deterministic hash embedding.

All computation runs on CPU — no GPU required.

Key functions:
  - build_faiss_index(chunks): creates and returns a VectorStore object
  - VectorStore.search(query, k=4): returns top-k relevant chunks
"""

import numpy as np
import faiss
import hashlib
import re
from typing import Any

# Load the embedding model once at module level.
# all-MiniLM-L6-v2 is fast, CPU-friendly, and produces 384-dim vectors.
_EMBED_MODEL = None


class _HashingEmbedder:
    """Small dependency-free embedder used as a safe fallback."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._token_pattern = re.compile(r"[a-zA-Z0-9]+")

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._token_pattern.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            token_hash = int.from_bytes(digest, byteorder="little", signed=False)
            idx = token_hash % self.dim
            sign = 1.0 if ((token_hash >> 1) & 1) == 0 else -1.0
            vec[idx] += sign
        return vec

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        del batch_size, show_progress_bar  # kept for API compatibility
        matrix = np.vstack([self._embed_one(t) for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        return matrix.astype(np.float32)


def get_embedding_model() -> Any:
    """Lazy-load the embedding model (loads once, reused across calls)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBED_MODEL = _HashingEmbedder(dim=384)
    return _EMBED_MODEL


class VectorStore:
    """
    Wraps a FAISS flat (exact) index with source metadata.

    Attributes:
        index: faiss.IndexFlatL2 — stores L2-distance embeddings
        chunks: list of dicts {'text': ..., 'source': ...}
        model: embedding model implementing .encode()
    """

    def __init__(self, chunks: list[dict]):
        """
        Build the FAISS index from a list of chunk dicts.

        Args:
            chunks: [{'text': '...', 'source': 'file.pdf'}, ...]
        """
        self.chunks = chunks
        self.model = get_embedding_model()

        # Extract plain text for embedding
        texts = [c["text"] for c in chunks]

        # Encode all chunks — batch encoding is faster than one-by-one
        # normalize_embeddings=True makes cosine similarity ≈ L2 search
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # FAISS requires float32
        embeddings = np.array(embeddings, dtype=np.float32)

        # Dimension of the embedding space (384 for MiniLM)
        dim = embeddings.shape[1]

        # IndexFlatIP = inner product (cosine) search after normalization
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """
        Find the top-k most semantically similar chunks for a query.

        Args:
            query: User's natural language question.
            k: Number of results to return (default 4).

        Returns:
            List of chunk dicts sorted by relevance (most relevant first).
        """
        # Encode the query with the same normalization
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_vec = np.array(query_vec, dtype=np.float32)

        # FAISS search returns distances and indices
        k = min(k, len(self.chunks))  # guard against tiny indexes
        distances, indices = self.index.search(query_vec, k)

        results = []
        for idx in indices[0]:
            if idx != -1:  # FAISS returns -1 for unfilled slots
                results.append(self.chunks[idx])

        return results


def build_faiss_index(chunks: list[dict]) -> VectorStore:
    """
    Convenience factory: build and return a VectorStore from chunks.

    Args:
        chunks: Output of pdf_processor.process_uploaded_pdfs()

    Returns:
        VectorStore instance ready for .search() calls.
    """
    return VectorStore(chunks)
