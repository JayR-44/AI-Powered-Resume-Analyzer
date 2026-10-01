"""Parse uploaded PDFs while retaining distinguishable source names."""

from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.documents import Document

from resume_checker.pdf_loader import load_pdf


def load_uploads(files: list[tuple[str, bytes]]) -> list[Document]:
    """Read uploads from temporary files, removing them after extraction."""
    if not files:
        raise ValueError("Upload at least one PDF.")
    pages = []
    with TemporaryDirectory(prefix="resume-checker-") as folder:
        for number, (name, content) in enumerate(files, start=1):
            safe_name = name.replace("\\", "/").rsplit("/", 1)[-1]
            if not safe_name.lower().endswith(".pdf"):
                raise ValueError(f"{safe_name}: please upload a PDF.")
            path = Path(folder) / f"{number}.pdf"
            path.write_bytes(content)
            try:
                document_pages = load_pdf(path)
            except ValueError as exc:
                raise ValueError(f"{safe_name}: {exc}") from exc
            for page in document_pages:
                page.metadata["source"] = safe_name
                page.metadata["document_id"] = str(number)
                page.metadata["display_source"] = f"CV {number}: {safe_name}"
            pages.extend(document_pages)
    return pages
