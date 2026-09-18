# Serve the frontend static app (separate from the API process).
from __future__ import annotations

import http.server
import socketserver
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
HOST = "127.0.0.1"
PORT = 5173


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)


if __name__ == "__main__":
    if not FRONTEND_DIR.is_dir():
        raise SystemExit(f"frontend directory not found: {FRONTEND_DIR}")
    with socketserver.TCPServer((HOST, PORT), QuietHandler) as httpd:
        print(f"Frontend → http://{HOST}:{PORT}/  (API expected at http://localhost:8000/api)")
        httpd.serve_forever()
