"""Local semantic search over PDF chunks."""

from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def build_vector_store(chunks: list[Document]) -> InMemoryVectorStore:
    """Embed chunks locally; download and cache model weights on first use."""
    if not chunks:
        raise ValueError("No text chunks to index.")
    cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "fastembed"
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=str(cache_dir),
        threads=2,
    )
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def search_chunks(
    store: InMemoryVectorStore, question: str, k: int = 3
) -> list[Document]:
    """Return the closest chunks, retaining their citation metadata."""
    if not question.strip():
        raise ValueError("Question must not be blank.")
    if k <= 0:
        raise ValueError("The number of results must be greater than zero.")
    return store.similarity_search(question.strip(), k=k)
