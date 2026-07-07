"""
PrivateRAG core — retrieval-augmented generation where nothing leaves the box.

Every stage is local: embeddings (nomic-embed-text) and generation (qwen2.5)
both run on a self-hosted Ollama server. There is deliberately no cloud
fallback in this module — the whole point is a hard guarantee that documents
and questions never touch a third-party API.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx
import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen2.5:3b"  # override per-request; 7b-q4 recommended on 8GB+ GPUs

CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200
TOP_K = 4
MIN_SIM = 0.35  # nomic-embed cosine floor — below this we refuse to answer

SYSTEM_PROMPT = (
    "You are PrivateRAG, a document-grounded assistant running fully on-premise. "
    "Answer ONLY from the provided context. If the context does not contain the "
    "answer, say you don't know — never invent facts. Be concise and cite the "
    "source titles you used."
)


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    vector: Optional[np.ndarray] = None


@dataclass
class Store:
    """In-memory vector store. Simple by design: the demo indexes dozens of
    documents, not millions — numpy cosine over a matrix is instant and keeps
    the dependency list honest (no vector-DB service to explain away)."""

    chunks: list[Chunk] = field(default_factory=list)
    _matrix: Optional[np.ndarray] = None

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        vecs = [c.vector for c in self.chunks]
        self._matrix = np.vstack(vecs) if vecs else None

    def search(self, qvec: np.ndarray, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self.chunks:
            return []
        sims = self._matrix @ qvec  # vectors are L2-normalised → dot = cosine
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] >= MIN_SIM]


def split_text(text: str) -> list[str]:
    """Paragraph-aware splitter: pack whole paragraphs up to CHUNK_CHARS,
    falling back to a hard split (with overlap) for monster paragraphs."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > CHUNK_CHARS:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(p), CHUNK_CHARS - CHUNK_OVERLAP):
                out.append(p[i : i + CHUNK_CHARS])
            continue
        if len(buf) + len(p) + 2 > CHUNK_CHARS and buf:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


async def embed(texts: list[str], client: httpx.AsyncClient) -> np.ndarray:
    """Batch-embed via the local Ollama server; L2-normalise for cosine-by-dot."""
    res = await client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    res.raise_for_status()
    vecs = np.array(res.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)


async def index_document(title: str, text: str, store: Store, client: httpx.AsyncClient) -> int:
    doc_id = hashlib.sha256(f"{title}:{len(text)}".encode()).hexdigest()[:8]
    pieces = split_text(text)
    if not pieces:
        return 0
    vecs = await embed(pieces, client)
    store.add([
        Chunk(doc_id=doc_id, title=title, text=p, vector=v)
        for p, v in zip(pieces, vecs)
    ])
    return len(pieces)


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{c.title}]\n{c.text}" for c, _ in hits)


async def generate_stream(
    question: str, context: str, client: httpx.AsyncClient, model: str = GEN_MODEL
) -> AsyncIterator[str]:
    """Stream a grounded answer token-by-token from the local model."""
    async with client.stream(
        "POST",
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": True,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        },
        timeout=300,
    ) as res:
        res.raise_for_status()
        async for line in res.aiter_lines():
            if not line.strip():
                continue
            data = json.loads(line)
            delta = (data.get("message") or {}).get("content", "")
            if delta:
                yield delta
            if data.get("done"):
                return


async def extract_structured(
    text: str, schema: dict, client: httpx.AsyncClient, model: str = GEN_MODEL
) -> dict:
    """Structured extraction with a JSON-schema-constrained local model —
    Ollama's `format` parameter enforces the schema at the sampler level, so
    the output parses every time (no retry loops, no regex repair)."""
    res = await client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": "Extract the requested fields from the document. Use null for fields not present.",
                },
                {"role": "user", "content": text},
            ],
        },
        timeout=300,
    )
    res.raise_for_status()
    return json.loads(res.json()["message"]["content"])
