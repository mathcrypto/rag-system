#!/usr/bin/env bash
# Ingest: wipe vector DB and rebuild from data/raw (or RAW_DIR).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

RAW_DIR="${RAW_DIR:-${ROOT}/data/raw}"
PERSIST_DIR="${PERSIST_DIR:-${ROOT}/data/vectordb}"

if [[ ! -d "$RAW_DIR" ]]; then
  echo "error: raw directory not found: $RAW_DIR" >&2
  exit 1
fi

echo "Clearing index at $PERSIST_DIR"
rm -rf "${PERSIST_DIR:?}"/*

echo "Building index from $RAW_DIR"
RAW_DIR="$RAW_DIR" PERSIST_DIR="$PERSIST_DIR" "$PYTHON" scripts/build_index.py

echo "Ingest complete."
