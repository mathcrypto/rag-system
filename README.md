# RAG System

LangChain-style RAG: load → split → embed (OpenAI) → Chroma → retrieve → (optional rerank) → generate.

Supports **dense**, **BM25**, **multi-query**, **hybrid** retrieval, **LLM routing**, and **Cohere reranking**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # set OPENAI_API_KEY (and COHERE_API_KEY for rerank)
```

## Data

Put `.txt` / `.md` / `.pdf` files in `data/raw/`.

Optional — download a small [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) slice:

```bash
pip install datasets
python -c "
from datasets import load_dataset
from pathlib import Path
out = Path('data/raw'); out.mkdir(parents=True, exist_ok=True)
ds = load_dataset('rajpurkar/squad', split='train[:80]')
seen, n = set(), 0
for row in ds:
    t = (row.get('context') or '').strip()
    if t and t not in seen:
        seen.add(t); (out / f'squad_{n:03d}.txt').write_text(t); n += 1
print(n, 'files')
"
```

## Index

```bash
rm -rf data/vectordb/*
python scripts/build_index.py
```

## Ask

**Routing** — an LLM picks `dense` | `bm25` | `multi_query` | `hybrid` for each question:

```bash
python -c "from generation.llm_client import answer; print(answer('What is the Grotto at Notre Dame?', use_routing=True))"
```

Force a strategy, or add rerank:

```bash
python -c "from generation.llm_client import answer; print(answer('What is the Grotto at Notre Dame?', strategy='hybrid'))"
python -c "from generation.llm_client import answer; print(answer('What is the Grotto at Notre Dame?', use_routing=True, use_rerank=True))"
```

See the chosen route:

```bash
python -c "from retrieval.router import retrieve_routed; s,d=retrieve_routed('What is the Grotto at Notre Dame?'); print(s, len(d))"
```

Ask about topics that appear in your `data/raw/` files.

## Layout

`src/ingestion` · `embedding` · `retrieval` (incl. `router.py`) · `reranking` · `generation` · `config`
