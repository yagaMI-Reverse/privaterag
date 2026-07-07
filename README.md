# PrivateRAG

**Retrieval-augmented generation where nothing leaves the machine.**

PrivateRAG is a document Q&A + structured-extraction engine that runs *entirely*
on local hardware. Embeddings, retrieval and generation all happen on a
self-hosted [Ollama](https://ollama.com) server — there are no API keys, no cloud
calls, and no data egress. The UI proves it with a live counter that stays at
`0 external requests` for the whole session.

Built for teams that can't send confidential documents (legal, medical, finance,
internal HR) to a third-party AI API but still want ChatGPT-style answers grounded
in their own files.

- **Chat** — ask questions, get streamed answers grounded in your documents, with
  cited sources and a hard refusal when the answer isn't in the corpus.
- **Structured extract** — pull typed JSON from any document using a JSON-Schema
  that is enforced at the model's sampler level (Ollama `format`), so the output
  parses every time — no retry loops, no "please respond in JSON" prompting.

---

## Why local

| Concern | Cloud LLM API | PrivateRAG |
|---|---|---|
| Confidential docs leave your network | Yes | **Never** |
| Per-token cost | Metered | **$0** |
| Works offline / air-gapped | No | **Yes** |
| Vendor lock-in | Yes | Swap models freely |
| Runs on | Someone else's GPU | **A ~$300 consumer GPU** |

---

## Benchmarks

Measured end-to-end on a single **NVIDIA RTX 4060 (8 GB)**, warm model:

| Model | First token | Generation | Structured extract | Notes |
|---|---|---|---|---|
| `qwen2.5:3b` (q4) | ~0.2 s | **~95 tok/s** | ~2.0 s | fast default |
| `qwen2.5:7b` (q4) | ~0.25 s | ~46 tok/s | ~3.5 s | higher quality |

Retrieval (local `nomic-embed-text` embeddings + cosine): **< 150 ms** per query.
Extraction accuracy verified against a schema (invoice → typed JSON): line items,
totals and dates extracted correctly on both models.

---

## How it works

```
        ┌─────────────────────  your machine  ─────────────────────┐
        │                                                           │
 docs ──┼──► chunk ──► nomic-embed-text ──► vector store (numpy)     │
        │                                        │                  │
 query ─┼──► nomic-embed-text ──► cosine top-k ──┘                  │
        │                             │                             │
        │                             ▼                             │
        │                    qwen2.5 (Ollama)  ──► streamed answer  │
        │                                          + cited sources  │
        └───────────────────────────────────────────────────────────┘
                         no network egress — ever
```

1. **Ingest** — documents are split paragraph-aware into ~1.4k-char chunks.
2. **Embed** — each chunk is embedded locally with `nomic-embed-text` and stored
   in an in-memory numpy matrix (L2-normalised for cosine-by-dot).
3. **Retrieve** — the query is embedded the same way; top-k chunks above a
   similarity floor become the context. Below the floor, the bot refuses instead
   of hallucinating.
4. **Generate** — `qwen2.5` streams a grounded answer over Ollama's chat API,
   citing the source titles it used.

---

## Run it

Requires [Ollama](https://ollama.com) and Python 3.10+.

```bash
# 1. pull the models (one-time)
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. start the engine
cd engine
pip install -r requirements.txt
uvicorn server:app --port 8017

# 3. open http://localhost:8017
```

Click **Load sample document** to index a bundled handbook, then ask the suggested
questions — or drop your own PDF / TXT / MD. Watch the `EXTERNAL REQUESTS` counter
in the header: it never moves.

---

## Stack

- **Ollama** — local model server (`qwen2.5:3b` / `7b`, `nomic-embed-text`)
- **FastAPI** — SSE streaming, upload, structured-extract endpoints
- **numpy** — in-memory cosine vector store (no vector-DB service to run)
- **Vanilla HTML/CSS/JS** — zero-build single-page UI

No LangChain, no vector database, no cloud SDK — the dependency list is honest and
small on purpose.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /docs/upload` | index a PDF / TXT / MD file |
| `POST /docs/seed` | index the bundled sample document |
| `POST /ask` | SSE-streamed grounded answer + sources |
| `POST /extract` | schema-enforced structured JSON extraction |
| `GET /health` | Ollama status + chunk count |
| `GET /models` | locally available generation models |

---

*Part of the [yagaMI-Reverse](https://github.com/yagaMI-Reverse) portfolio.*
