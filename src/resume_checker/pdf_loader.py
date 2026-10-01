"""Load local PDFs into one LangChain Document per page."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """Extract page text while preserving source and page metadata."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a file with a .pdf extension.")

    try:
        pages = PyPDFLoader(str(path), mode="page").load()
    except Exception as exc:
        raise ValueError(
            "Could not read the PDF. Check that it is valid, readable, "
            "and not password protected."
        ) from exc

    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            "No text was extracted. The PDF may be blank or scanned; "
            "scanned PDFs need OCR, which is not implemented yet."
        )
    return pages
