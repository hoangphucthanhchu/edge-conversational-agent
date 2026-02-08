"""RAG: ingest docs from data/rag/, chunk, embed (sentence-transformers), ChromaDB, retrieve."""

from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.config import Settings
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
    """Ingest docs, embed, store in ChromaDB; retrieve top-k chunks as context string."""

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "data/rag/chroma_db",
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
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection_name = "rag_docs"
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, docs_dir: str | Path) -> int:
        """Read .txt/.md from docs_dir, chunk, embed, add to ChromaDB. Returns num chunks."""
        docs_dir = Path(docs_dir)
        if not docs_dir.is_dir():
            return 0
        chunks_all: list[str] = []
        ids_all: list[str] = []
        for path in sorted(docs_dir.rglob("*")):
            if path.suffix.lower() not in (".txt", ".md"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
            for i, c in enumerate(chunks):
                chunks_all.append(c)
                ids_all.append(f"{path.name}_{i}")
        if not chunks_all:
            return 0
        embeddings = self._encoder.encode(chunks_all).tolist()
        self._collection.upsert(
            ids=ids_all,
            documents=chunks_all,
            embeddings=embeddings,
        )
        return len(chunks_all)

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """Return context string from top-k chunks for query."""
        k = top_k if top_k is not None else self.top_k
        if not query or not query.strip():
            return ""
        n = self._collection.count()
        if n == 0:
            return ""
        q_emb = self._encoder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(k, n),
        )
        docs = results.get("documents") or [[]]
        if not docs or not docs[0]:
            return ""
        return "\n\n".join(docs[0])

    def clear(self) -> None:
        """Remove all documents from the collection (for rebuild)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
