"""Embedding retrieval: nomic-embed-text (via Ollama) + a local Chroma vector database.

Drop-in alternative to the TF-IDF search in retrieve.py. Every search returns the
same list of (score, Chunk) that the rest of the pipeline already uses, so verdicts,
citations and the chat do not change.

How it works
1. Embed: each chunk's heading + text is sent to Ollama's nomic-embed-text model,
   which returns a 768-number vector that captures its meaning.
2. Store: vectors are saved in Chroma (a folder on disk) with the document code,
   heading and pages as metadata, so a search can be limited to one document.
3. Retrieve: the test or question is embedded the same way and Chroma returns the
   closest chunks (cosine similarity, 1.0 = same meaning).

The database is kept in sync with the loaded corpus: only new or changed chunks
are embedded, and chunks that are no longer in the corpus are removed. A first
run embeds all 270 chunks; later runs reuse them.
"""
import hashlib
import re

import requests

# nomic-embed-text expects a task prefix on every input. Skipping it quietly
# lowers search quality.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BATCH_SIZE = 32

DEFAULTS = {
    "model": "nomic-embed-text",
    "host": "http://localhost:11434",
    "db_dir": "data/chroma_db",
    "timeout_seconds": 120,
}


def embedding_settings(cfg):
    """Embedding settings from config.yaml, with defaults for anything missing."""
    settings = dict(DEFAULTS)
    settings.update((cfg or {}).get("embedding") or {})
    return settings


def chunk_id(chunk):
    """Stable id from the chunk's content, so unchanged chunks are never re-embedded."""
    raw = f"{chunk.doc_code}\0{chunk.page_start}\0{chunk.page_end}\0{chunk.heading}\0{chunk.text}"
    return hashlib.sha1(raw.encode("utf8")).hexdigest()


class OllamaEmbedder:
    def __init__(self, model, host, timeout=120):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def embed(self, texts):
        r = requests.post(f"{self.host}/api/embed", timeout=self.timeout,
                          json={"model": self.model, "input": list(texts)})
        if r.status_code == 404 and "not found" in r.text.lower():
            raise RuntimeError(f"Embedding model '{self.model}' is not installed. Run: ollama pull {self.model}")
        r.raise_for_status()
        return r.json()["embeddings"]


class VectorIndex:
    """Chroma-backed index with the same `retrieve(query, top_k)` as retrieve.CorpusIndex."""

    def __init__(self, docs, cfg=None, embedder=None, client=None):
        import chromadb  # imported here so TF-IDF users do not need chromadb installed

        s = embedding_settings(cfg)
        self.embedder = embedder or OllamaEmbedder(s["model"], s["host"], s["timeout_seconds"])
        self.client = client or chromadb.PersistentClient(path=s["db_dir"])
        # One collection per embedding model: vectors from different models cannot be mixed.
        name = "chunks_" + re.sub(r"[^A-Za-z0-9_-]+", "-", s["model"]).strip("-_")
        self.collection = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self.chunks = {chunk_id(c): c for d in docs for c in d.chunks}
        self.added, self.removed = self._sync()

    def _sync(self):
        existing = set(self.collection.get(include=[])["ids"])
        stale = sorted(existing - set(self.chunks))
        if stale:
            self.collection.delete(ids=stale)
        missing = [i for i in self.chunks if i not in existing]
        for start in range(0, len(missing), BATCH_SIZE):
            ids = missing[start:start + BATCH_SIZE]
            chunks = [self.chunks[i] for i in ids]
            texts = [f"{c.heading}\n{c.text}" for c in chunks]
            self.collection.add(
                ids=ids,
                embeddings=self.embedder.embed([DOC_PREFIX + t for t in texts]),
                documents=texts,
                metadatas=[{"doc_code": c.doc_code, "heading": c.heading,
                            "page_start": c.page_start, "page_end": c.page_end} for c in chunks],
            )
        return len(missing), len(stale)

    def search(self, query, top_k, doc_codes=None):
        """Top-k chunks for a query, optionally limited to some document codes."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query.strip() or not self.chunks:
            return []
        where = None
        if doc_codes:
            codes = sorted(set(doc_codes))
            where = {"doc_code": codes[0]} if len(codes) == 1 else {"doc_code": {"$in": codes}}
        vector = self.embedder.embed([QUERY_PREFIX + query])[0]
        res = self.collection.query(query_embeddings=[vector], n_results=top_k, where=where,
                                    include=["distances"])
        return [(1.0 - dist, self.chunks[i])
                for i, dist in zip(res["ids"][0], res["distances"][0]) if i in self.chunks]

    def retrieve(self, query, top_k):
        """Across the whole corpus (used by the Conversation tab and chat CLI)."""
        return self.search(query, top_k)

    def retrieve_per_document(self, query, docs, top_k):
        """Top-k chunks PER document, matching retrieve.retrieve (used by the test runner)."""
        out = []
        for d in docs:
            hits = self.search(query, top_k, [d.code])
            if not hits and d.chunks:
                hits = [(0.0, d.chunks[0])]
            out += hits
        return out
