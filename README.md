# RAG System — end-to-end service

LangChain-style RAG: **load → split → embed → Chroma → retrieve → (rerank) → generate**.

**Backend** (`backend/`) and **frontend** (`frontend/`) are separate — matches the online RAG + offline ingest split.

Supports **dense**, **BM25**, **multi-query**, **RAG-Fusion** (RRF), **hybrid**, **LLM routing**, and **Cohere reranking**.

## Lifecycle

```text
data/raw  →  ingest / build_index  →  data/vectordb          (offline)
browser   →  frontend (:5173)  →  API (:8000/api)  →  ask()  (online)
```

### 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # OPENAI_API_KEY; COHERE_API_KEY; CORS_ORIGINS
```

### 2. Data + ingest

```bash
# put files in data/raw/
./scripts/ingest_documents.sh
python scripts/run_eval.py
```

### 3. Run API + frontend (two terminals)

```bash
python scripts/run_api.py        # http://localhost:8000  →  /api/health, /api/ask, /docs
python scripts/run_frontend.py   # http://localhost:5173/
```

Open **http://localhost:5173/** for chat. The UI calls `http://localhost:8000/api/...`.

```bash
curl -s http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the Grotto at Notre Dame?","strategy":"dense"}'
```

## Layout

| Path | Role |
|------|------|
| `frontend/` | HTML · CSS · JS (chat client) |
| `backend/api/` | FastAPI (`/api/health`, `/api/ask`) |
| `backend/ingestion` · `embedding` · `retrieval` · `reranking` · `generation` | RAG pipeline |
| `scripts/run_api.py` | Backend HTTP process |
| `scripts/run_frontend.py` | Frontend static server |
| `scripts/ingest_documents.sh` · `build_index.py` | Offline indexing |
| `scripts/run_eval.py` | Retrieval smoke |
| `evaluation/` | Eval helpers |

Override API URL in the browser console if needed: `window.RAG_API_BASE = "https://your-api/api"`.
