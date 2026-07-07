"""
PrivateRAG API — FastAPI wrapper around the local-only RAG core.

Run:  uvicorn server:app --port 8017
Everything (embeddings, generation, this server) stays on your machine.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import rag

STATIC_DIR = Path(__file__).parent / "static"

# Swagger moved to /api-docs — GET /docs is our document-listing endpoint.
app = FastAPI(title="PrivateRAG", version="0.1.0", docs_url="/api-docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STORE = rag.Store()
HTTP = httpx.AsyncClient()
QUESTIONS: list[dict] = []


class AskIn(BaseModel):
    question: str
    model: str = rag.GEN_MODEL


class ExtractIn(BaseModel):
    text: str
    schema_: dict
    model: str = rag.GEN_MODEL


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/models")
async def models():
    """Local models available for generation (from the Ollama server)."""
    res = await HTTP.get(f"{rag.OLLAMA_URL}/api/tags", timeout=5)
    res.raise_for_status()
    out = [
        {"name": m["name"], "gb": round(m.get("size", 0) / 1e9, 1)}
        for m in res.json().get("models", [])
        if "embed" not in m["name"]
    ]
    return {"models": out, "default": rag.GEN_MODEL}


@app.get("/docs")
def list_docs():
    seen: dict[str, int] = {}
    for c in STORE.chunks:
        seen[c.title] = seen.get(c.title, 0) + 1
    return {"docs": [{"title": t, "chunks": n} for t, n in seen.items()]}


SAMPLE_DOC = (STATIC_DIR / "sample_handbook.txt")


@app.post("/docs/seed")
async def seed():
    """Index the bundled sample document — lets the demo work in one click."""
    if any(c.title == "Acme Handbook (sample)" for c in STORE.chunks):
        return {"ok": True, "already": True}
    n = await rag.index_document(
        "Acme Handbook (sample)", SAMPLE_DOC.read_text(encoding="utf-8"), STORE, HTTP
    )
    return {"ok": True, "chunks": n}


@app.get("/health")
async def health():
    try:
        res = await HTTP.get(f"{rag.OLLAMA_URL}/api/version", timeout=3)
        ollama = res.json().get("version")
    except Exception:
        ollama = None
    return {
        "ok": ollama is not None,
        "ollama": ollama,
        "chunks": len(STORE.chunks),
        "local_only": True,
    }


@app.post("/docs/upload")
async def upload(file: UploadFile):
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(413, "5 MB limit")
    name = (file.filename or "document").rsplit(".", 1)[0]
    if (file.filename or "").lower().endswith(".pdf"):
        from io import BytesIO

        from pypdf import PdfReader

        text = "\n\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(raw)).pages)
    else:
        text = raw.decode("utf-8", errors="replace")
    n = await rag.index_document(name, text, STORE, HTTP)
    if n == 0:
        raise HTTPException(400, "No extractable text")
    return {"ok": True, "title": name, "chunks": n}


@app.post("/ask")
async def ask(body: AskIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    t0 = time.perf_counter()
    qvec = (await rag.embed([q], HTTP))[0]
    hits = STORE.search(qvec)
    retrieval_ms = round((time.perf_counter() - t0) * 1000)

    record = {"id": str(uuid.uuid4())[:8], "q": q, "ts": time.time()}

    async def gen():
        sources = [{"title": c.title, "score": round(s, 3)} for c, s in hits]
        meta = {"sources": sources, "retrieval_ms": retrieval_ms, "grounded": bool(hits)}
        yield f"data: {json.dumps(meta)}\n\n"
        if not hits:
            msg = "I couldn't find anything relevant in the indexed documents."
            record.update(answer=msg, grounded=False)
            QUESTIONS.append(record)
            yield f"data: {json.dumps({'delta': msg})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
        collected = ""
        t_first = None  # measure tok/s from the first token so a cold model load doesn't skew it
        ntok = 0
        async for delta in rag.generate_stream(q, rag.build_context(hits), HTTP, body.model):
            if t_first is None:
                t_first = time.perf_counter()
            collected += delta
            ntok += 1
            yield f"data: {json.dumps({'delta': delta})}\n\n"
        gen_s = time.perf_counter() - (t_first or time.perf_counter())
        record.update(answer=collected, grounded=True)
        QUESTIONS.append(record)
        yield f"data: {json.dumps({'done': True, 'gen_tokens': ntok, 'tok_per_s': round(ntok / max(gen_s, 0.001), 1)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/extract")
async def extract(body: ExtractIn):
    t0 = time.perf_counter()
    data = await rag.extract_structured(body.text, body.schema_, HTTP, body.model)
    return {"ok": True, "data": data, "ms": round((time.perf_counter() - t0) * 1000)}


@app.get("/questions")
def questions(limit: int = 50):
    return list(reversed(QUESTIONS[-limit:]))
