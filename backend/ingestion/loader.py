# Loads PDFs, URLs, docs from any source
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from langchain_core.documents import Document

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SAMPLE_PATH = RAW_DIR / "sample.txt"
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".markdown"}
PDF_EXTENSION = ".pdf"


def _detect_mime(path: Path) -> str:
    try:
        import magic

        return magic.from_file(str(path), mime=True)
    except Exception:
        return {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".markdown": "text/markdown",
            ".pdf": "application/pdf",
        }.get(path.suffix.lower(), "application/octet-stream")


def _base_metadata(source: str, mime: str, **extra) -> dict:
    return {
        "source": source,
        "mime_type": mime,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata=_base_metadata(
                source=str(path),
                mime=_detect_mime(path),
                filename=path.name,
            ),
        )
    ]


def _load_pdf(path: Path) -> list[Document]:
    try:
        from docling.document_converter import DocumentConverter

        result = DocumentConverter().convert(str(path))
        text = result.document.export_to_markdown()
        return [
            Document(
                page_content=text,
                metadata=_base_metadata(
                    source=str(path),
                    mime="application/pdf",
                    filename=path.name,
                    parser="docling",
                ),
            )
        ]
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs: list[Document] = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            docs.append(
                Document(
                    page_content=page_text,
                    metadata=_base_metadata(
                        source=str(path),
                        mime="application/pdf",
                        filename=path.name,
                        parser="pypdf",
                        page_number=i + 1,
                    ),
                )
            )
        if not docs:
            raise ValueError(f"No text found in {path}")
        return docs


def _load_url(url: str) -> list[Document]:
    with urlopen(url, timeout=30) as response:
        raw = response.read()
        content_type = response.headers.get_content_type()
    text = raw.decode("utf-8", errors="replace")
    return [
        Document(
            page_content=text,
            metadata=_base_metadata(
                source=url,
                mime=content_type,
                filename=Path(urlparse(url).path).name or url,
            ),
        )
    ]


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    mime = _detect_mime(path)
    if suffix in TEXT_EXTENSIONS or mime.startswith("text/"):
        return _load_text(path)
    if suffix == PDF_EXTENSION or mime == "application/pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {path} (mime: {mime})")


def load_directory(directory: str | Path = RAW_DIR) -> list[Document]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            docs.extend(load_file(path))
        except ValueError:
            continue
    return docs


def load_sample() -> Document:
    return load_file(SAMPLE_PATH)[0]
