"""Prepare page documents for embedding and retrieval."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_pages(
    pages: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Split within each page; sizes are characters, not tokens."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Chunk overlap must be non-negative and less than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(
        [page for page in pages if page.page_content.strip()]
    )
