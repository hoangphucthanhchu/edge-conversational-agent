"""RAG: ingest docs from data/rag/, chunk, embed (sentence-transformers), FAISS, retrieve."""

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks (by characters)."""
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap if overlap < chunk_size else end
    return chunks


class RAGRetriever:
    """Ingest docs, embed, store in FAISS; retrieve top-k chunks as context string."""

    INDEX_FILENAME = "index.faiss"
    CHUNKS_FILENAME = "chunks.pkl"

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "data/rag/faiss_index",
        top_k: int = 3,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.embed_model_name = embed_model
        self.persist_dir = Path(persist_dir)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = SentenceTransformer(embed_model)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[str] = []

    def _index_path(self) -> Path:
        return self.persist_dir / self.INDEX_FILENAME

    def _chunks_path(self) -> Path:
        return self.persist_dir / self.CHUNKS_FILENAME

    def _load_index(self) -> bool:
        """Load FAISS index and chunks from disk if present. Returns True if loaded."""
        ip, cp = self._index_path(), self._chunks_path()
        if not ip.exists() or not cp.exists():
            return False
        self._index = faiss.read_index(str(ip))
        with open(cp, "rb") as f:
            self._chunks = pickle.load(f)
        return True

    def _save_index(self) -> None:
        """Save FAISS index and chunks to disk."""
        if self._index is None or not self._chunks:
            return
        faiss.write_index(self._index, str(self._index_path()))
        with open(self._chunks_path(), "wb") as f:
            pickle.dump(self._chunks, f)

    def build_index(self, docs_dir: str | Path) -> int:
        """Read .txt/.md from docs_dir, chunk, embed, add to FAISS. Returns num chunks."""
        docs_dir = Path(docs_dir)
        if not docs_dir.is_dir():
            return 0
        chunks_all: list[str] = []
        for path in sorted(docs_dir.rglob("*")):
            if path.suffix.lower() not in (".txt", ".md"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for c in _chunk_text(text, self.chunk_size, self.chunk_overlap):
                chunks_all.append(c)
        if not chunks_all:
            return 0
        embeddings = self._encoder.encode(chunks_all, convert_to_numpy=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunks = chunks_all
        self._save_index()
        return len(chunks_all)

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """Return context string from top-k chunks for query."""
        k = top_k if top_k is not None else self.top_k
        if not query or not query.strip():
            return ""
        if self._index is None and not self._load_index():
            return ""
        n = self._index.ntotal
        if n == 0 or not self._chunks:
            return ""
        k = min(k, n)
        q_emb = self._encoder.encode([query], convert_to_numpy=True)
        if not isinstance(q_emb, np.ndarray):
            q_emb = np.array(q_emb, dtype=np.float32)
        if q_emb.dtype != np.float32:
            q_emb = q_emb.astype(np.float32)
        faiss.normalize_L2(q_emb)
        _, indices = self._index.search(q_emb, k)
        hits = [self._chunks[i] for i in indices[0] if 0 <= i < len(self._chunks)]
        return "\n\n".join(hits) if hits else ""

    def clear(self) -> None:
        """Remove index and chunks (for rebuild)."""
        self._index = None
        self._chunks = []
        self._index_path().unlink(missing_ok=True)
        self._chunks_path().unlink(missing_ok=True)
