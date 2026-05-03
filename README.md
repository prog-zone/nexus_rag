# Nexus RAG

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688.svg?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-316192.svg?logo=postgresql)
![Qdrant](https://img.shields.io/badge/Qdrant-hybrid_search-red.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A backend API for AI-assisted legal document analysis. Lawyers and legal teams upload case documents, then query them through a chat interface — getting accurate, source-cited answers grounded strictly in their own files.

Built with FastAPI, Qdrant, and GPT-4o. Fully async, Dockerized, and production-ready.

---

## The Core Idea

Legal work involves reading through contracts, case files, and agreements looking for specific clauses, obligations, or precedents. Nexus lets you upload those documents into a **case** (a workspace), open a **chat**, and ask questions like:

> *"What does clause 9b say about termination?"*
> *"Compare the indemnity clauses across both contracts."*
> *"What are the payment obligations in the service agreement?"*

The system finds the relevant chunks, reranks them, and streams a grounded response — with mandatory source citations. If the answer isn't in the documents, it says so explicitly. It never guesses.

> *"The system is intentionally restricted to uploaded case documents to prevent hallucination of legal facts from unverified sources."*
---

## How the RAG Pipeline Works

Every message goes through this pipeline:

```
User Query
    │
    ▼
Query Understanding  ← GPT-4o extracts: query type, exact legal entity,
    │                  document hint, and sub-queries
    ▼
Hybrid Search (parallel, per sub-query)
    │   Dense vectors  →  VoyageAI voyage-law-2 (1024-dim, legal-optimized)
    │   Sparse vectors →  BM25 (FastEmbed)
    │   Fusion         →  Reciprocal Rank Fusion (RRF) in Qdrant
    │   Sources        →  case documents + inline pastes + chat memory
    ▼
Pre-filter → top-8 by score, deduplicated
    ▼
Reranking  ← VoyageAI rerank-2.5 with legal-domain instruction
    │         Relevance threshold: 0.3 (below this = no context found)
    ▼
GPT-4o (streamed via SSE)
    │   Strict system prompt: cite sources, never hallucinate,
    │   admit when the answer isn't in the documents
    ▼
Saved to DB + chat memory tracked for future retrieval
```

**Why hybrid search?** Dense vectors understand meaning but miss exact matches — clause numbers, party names, specific legal terms. BM25 catches those. RRF fusion combines both without needing to tune weights.

**Why VoyageAI over OpenAI embeddings?** `voyage-law-2` is trained specifically on legal text. Clause references and legal terminology embed more accurately.

**Why query decomposition?** A question like *"compare the termination clauses in contract A and B"* gets split into two sub-queries, each searched independently, then merged before reranking. A single query would miss one of them.

**Chat memory in Qdrant:** Once a conversation grows past ~6,000 estimated tokens, past exchanges get vectorized and pushed to a separate Qdrant collection. Future queries can then retrieve relevant past conversation context the same way they retrieve documents.

---

## Data Model

```
User
 └── Cases (workspaces, e.g. "Smith vs Jones")
      ├── Documents (uploaded files, stored in S3)
      └── Chats
           └── Messages (user + assistant turns, tracked for memory)
```

Documents have a lifecycle: `PENDING → PROCESSING → COMPLETED / FAILED`, managed by async background tasks.

---

## Tech Stack

| | |
|---|---|
| **API** | FastAPI (async), Python 3.13 |
| **Database** | PostgreSQL, SQLAlchemy (async), Alembic |
| **Vector DB** | Qdrant — two collections: documents + chat memory |
| **Embeddings** | VoyageAI voyage-law-2 (dense) + BM25/FastEmbed (sparse) |
| **Reranker** | VoyageAI rerank-2.5 |
| **LLM** | GPT-4o, streamed via SSE |
| **Document Parsing** | Unstructured API (hi-res OCR, chunked by title) |
| **File Storage** | S3-compatible (MinIO for local dev) |
| **Task Queue** | Taskiq + Redis (background ingestion + scheduled cleanup) |
| **Auth** | JWT + Argon2 + OTP email verification |
| **RAG Evaluation** | RAGAS (context precision, recall, faithfulness, answer relevancy) |

---

## Auth

- Access tokens (15 min) + refresh tokens (7 days), both as HttpOnly cookies
- Refresh tokens stored in the database — revoked on logout, password change, and account deletion
- Rotating refresh tokens: if a used token is replayed, all sessions for that user are immediately wiped (token theft detection)
- OTP-based email verification and password reset, hashed with SHA-256, expire in 15 minutes
- Argon2 password hashing (async, off the event loop)
- Rate limiting on sensitive endpoints — proxy-header-aware IP detection
- Security headers on every response: HSTS, X-Frame-Options, X-Content-Type-Options, XSS protection

---

## Document Ingestion Flow

```
POST /cases/{id}/documents
    │
    ├── File size check (configurable limit)
    ├── Doc count check per case (configurable limit)
    ├── Upload to S3
    ├── DB record created (status: PENDING)
    └── Background task queued (Taskiq → Redis)
            │
            ├── Fetch file content from S3
            ├── Unstructured API → hi-res parsing, chunked by title
            ├── Dense + sparse embeddings generated
            ├── Points upserted to Qdrant (documents collection)
            └── Status updated → COMPLETED or FAILED
```

Inline text pastes follow the same pipeline (with `AUTO` strategy) and land in the chat memory collection instead.

---

## Running Locally

```bash
git clone https://github.com/yourusername/nexus-rag.git
cd nexus-rag
cp .env.example .env   # fill in your API keys
```

Start infrastructure:
```bash
docker compose -f docker-compose.local.yml up -d
# Starts: PostgreSQL, Qdrant, Redis, MinIO, pgAdmin
```

Run the API:
```bash
uv sync
alembic upgrade head
fastapi dev app/main.py
```

Docs at `http://localhost:8000/docs`

**Production:**
```bash
docker compose up -d
# Runs migrations automatically, binds to 127.0.0.1:8000
```

---

## Tests

Tests run against an isolated `_test` database — created fresh before each run, dropped after.

```bash
pytest          # integration + unit tests
pytest -m ragas # RAG quality evaluation (slow, needs live API keys)
```

Test coverage: auth flows, user endpoints, full RAG endpoint suite (cases, documents, chats, messaging), JWT security (forged signatures, expired tokens), CORS, security headers.

RAGAS evaluation measures context precision, context recall, faithfulness, and answer relevancy — separately for retrieval-only and end-to-end pipeline.

---

## License

MIT