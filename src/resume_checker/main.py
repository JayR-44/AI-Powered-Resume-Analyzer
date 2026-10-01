"""Preview PDF chunks or ask questions with Groq."""

import argparse
import sys

from resume_checker.pdf_loader import load_pdf
from resume_checker.text_splitter import split_pages


def main() -> None:
    # Preserve Unicode PDF text and model output in Windows terminals and pipes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to a local PDF, for example data/cv.pdf")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum characters per chunk (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Target overlap in characters (default: 200)")
    parser.add_argument("--question", help="Generate a grounded answer to this question")
    parser.add_argument("--search-only", action="store_true", help="Retrieve passages without calling Groq")
    parser.add_argument("--top-k", type=int, default=3, help="Number of search results (default: 3)")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.question is not None and not args.question.strip():
        parser.error("--question must not be blank")
    if args.search_only and args.question is None:
        parser.error("--search-only requires --question")

    try:
        pages = load_pdf(args.pdf)
        chunks = split_pages(pages, args.chunk_size, args.chunk_overlap)
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")

    total_chars = sum(len(page.page_content) for page in pages)
    print(f"Loaded {len(pages)} page(s), {total_chars:,} characters.")
    if args.question is not None:
        from resume_checker.vector_store import build_vector_store, search_chunks

        llm = None
        if not args.search_only:
            from resume_checker.qa import answer_question, create_llm

            try:
                llm = create_llm()
            except ValueError as exc:
                parser.exit(1, f"Configuration error: {exc}\n")

        print(f"Embedding {len(chunks)} chunk(s) locally. First use downloads the model.", flush=True)
        try:
            store = build_vector_store(chunks)
            matches = search_chunks(store, args.question, args.top_k)
        except Exception as exc:
            parser.exit(1, f"Search failed: {exc}\n")
        if llm is not None:
            print("Generating an answer with Groq...", flush=True)
            try:
                answer = answer_question(llm, args.question, matches)
            except ValueError as exc:
                parser.exit(1, f"Answer failed: {exc}\n")
            except Exception:
                parser.exit(1, "Groq request failed. Check your API key, model access, quota, and network connection.\n")
            print(f"\nAnswer:\n{answer}")
        print(f"\nRetrieved passages ({len(matches)}):")
        for number, match in enumerate(matches, start=1):
            print(f"\n--- Result {number} | Page {match.metadata['page'] + 1} ---")
            print(f"Source: {match.metadata['source']}")
            print(match.page_content)
        return

    print(f"Created {len(chunks)} chunk(s). Showing up to 5 previews.")
    for number, chunk in enumerate(chunks[:5], start=1):
        page_number = chunk.metadata["page"] + 1
        print(f"\n--- Chunk {number} | Page {page_number} | {len(chunk.page_content)} characters ---")
        print(chunk.page_content[:300])
        if len(chunk.page_content) > 300:
            print("[Preview truncated to 300 characters]")


if __name__ == "__main__":
    main()
