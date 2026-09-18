# Run the production API (uvicorn). Index first with scripts/build_index.py.
from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
