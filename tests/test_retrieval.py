"""Integration checks using the real local embedding model (download on first run)."""

import contextlib
import io
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from resume_checker.main import main
from resume_checker.vector_store import build_vector_store, search_chunks


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = [
            Document(page_content="Developed Python APIs with Django and PostgreSQL for five years.", metadata={"source": "sample.pdf", "page": 0}),
            Document(page_content="Baked sourdough bread and decorated cakes in a bakery.", metadata={"source": "sample.pdf", "page": 1}),
            Document(page_content="Designed gardens and maintained trees and lawns.", metadata={"source": "sample.pdf", "page": 2}),
        ]
        cls.store = build_vector_store(cls.pages)

    def test_semantic_ranking_and_metadata(self):
        matches = search_chunks(self.store, "What software engineering experience is listed?", k=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].metadata, self.pages[0].metadata)
        self.assertEqual(matches[0].page_content, self.pages[0].page_content)
        self.assertEqual(len(search_chunks(self.store, "experience", k=10)), 3)

    def test_invalid_inputs(self):
        for query, k in [(" ", 3), ("skills", 0), ("skills", -1)]:
            with self.assertRaises(ValueError):
                search_chunks(self.store, query, k)
        with self.assertRaises(ValueError):
            build_vector_store([])

    def test_cli_search_and_preview(self):
        for extra_args, expected in [([], "Chunk 1 | Page 1"), (["--question", "software engineering", "--top-k", "1", "--search-only"], "Result 1 | Page 1")]:
            output = io.StringIO()
            with patch("sys.argv", ["resume-checker", "sample.pdf", *extra_args]), patch("resume_checker.main.load_pdf", return_value=self.pages), patch("resume_checker.vector_store.build_vector_store", return_value=self.store) as build, contextlib.redirect_stdout(output):
                main()
            self.assertIn(expected, output.getvalue())
            self.assertEqual(build.call_count, int(bool(extra_args)))


if __name__ == "__main__":
    unittest.main()
