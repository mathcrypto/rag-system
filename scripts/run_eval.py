# Run retrieval smoke eval (exit 0 on pass, non-zero on fail).
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/run_eval.py` without install quirks for evaluation/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evaluation.retrieval_eval import run_retrieval_smoke


def main() -> int:
    try:
        run_retrieval_smoke()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: dense retrieval smoke (Grotto / Notre Dame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
