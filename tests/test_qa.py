"""Offline checks for configuration and grounded-answer requests."""

import os
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from resume_checker.qa import answer_question, create_llm
from resume_checker.main import main


class AnswerTests(unittest.TestCase):
    def test_cli_answer(self):
        passages = [Document(page_content="Python", metadata={"page": 0, "source": "sample.pdf"})]
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="Python [Passage 1, page 1]")
        output = io.StringIO()
        with patch("sys.argv", ["resume-checker", "sample.pdf", "--question", "Skills?"]), patch("resume_checker.main.load_pdf", return_value=passages), patch("resume_checker.vector_store.build_vector_store"), patch("resume_checker.vector_store.search_chunks", return_value=passages), patch("resume_checker.qa.create_llm", return_value=llm), contextlib.redirect_stdout(output):
            main()
        self.assertIn("Answer:\nPython [Passage 1, page 1]", output.getvalue())
        self.assertIn("Source: sample.pdf", output.getvalue())

    def test_configuration_and_alias(self):
        with patch("resume_checker.qa.dotenv_values", return_value={}), patch.dict(os.environ, {"GROQ_API_KEY": "test-only", "GROQ_CHAT_MODEL": "alias-model"}, clear=True), patch("resume_checker.qa.ChatGroq") as constructor:
            create_llm()
            self.assertEqual(constructor.call_args.kwargs["model"], "alias-model")
            os.environ["LLM_MODEL"] = "preferred-model"
            create_llm()
            self.assertEqual(constructor.call_args.kwargs["model"], "preferred-model")

    def test_missing_key(self):
        with patch("resume_checker.qa.dotenv_values", return_value={}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                create_llm()

    def test_completion_settings(self):
        with patch.dict(os.environ, {}, clear=True), patch("resume_checker.qa.dotenv_values", return_value={"GROQ_API_KEY": "test-only"}) as settings, patch("resume_checker.qa.ChatGroq") as constructor:
            create_llm()
            self.assertEqual(constructor.call_args.kwargs["max_tokens"], 8192)
            self.assertEqual(constructor.call_args.kwargs["reasoning_effort"], "low")
            settings.return_value = {"GROQ_API_KEY": "test-only", "LLM_MAX_TOKENS": "4096", "LLM_REASONING_EFFORT": "medium"}
            create_llm()
            self.assertEqual(constructor.call_args.kwargs["max_tokens"], 4096)
            for value in ("zero", "0", "999999"):
                settings.return_value["LLM_MAX_TOKENS"] = value
                with self.assertRaisesRegex(ValueError, "LLM_MAX_TOKENS"):
                    create_llm()

    def test_env_edits_are_loaded_without_restart(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "old-key"}, clear=True), patch("resume_checker.qa.dotenv_values", side_effect=[{"GROQ_API_KEY": "new-key"}, {"GROQ_API_KEY": "updated-key"}]), patch("resume_checker.qa.ChatGroq") as constructor:
            create_llm()
            self.assertEqual(constructor.call_args.kwargs["api_key"], "new-key")
            create_llm()
            self.assertEqual(constructor.call_args.kwargs["api_key"], "updated-key")
            self.assertEqual(os.environ["GROQ_API_KEY"], "old-key")

    def test_prompt_and_response(self):
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="Python [Passage 1, page 3]")
        answer = answer_question(llm, "Which skills?", [Document(page_content="Python", metadata={"page": 2, "source": "private/local/cv.pdf"})])
        self.assertEqual(answer, "Python [Passage 1, page 3]")
        messages = llm.invoke.call_args.args[0]
        self.assertIn("untrusted", messages[0].content)
        self.assertIn("[Passage 1, page 3]\nPython", messages[1].content)
        self.assertNotIn("private/local", messages[1].content)

    def test_empty_evidence_and_empty_output(self):
        llm = Mock()
        self.assertIn("couldn't find", answer_question(llm, "Skills?", []))
        llm.invoke.assert_not_called()
        llm.invoke.return_value = AIMessage(content="")
        with self.assertRaisesRegex(ValueError, "no answer"):
            answer_question(llm, "Skills?", [Document(page_content="Python", metadata={"page": 0})])

    def test_truncated_output(self):
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="Partial", response_metadata={"finish_reason": "length"})
        with self.assertRaisesRegex(ValueError, "output limit"):
            answer_question(llm, "Skills?", [Document(page_content="Python", metadata={"page": 0})])


if __name__ == "__main__":
    unittest.main()
