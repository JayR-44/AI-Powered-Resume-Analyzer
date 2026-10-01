"""UI checks with real PDF extraction and retrieval, without Groq requests."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from groq import RateLimitError

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from streamlit.testing.v1 import AppTest


def pdf_upload(name, text):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode())
    page[NameObject('/Contents')] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return SimpleNamespace(name=name, getvalue=output.getvalue)


class StreamlitTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'), default_timeout=30)

    def test_empty(self):
        app = self.app().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(app.button(key='prepare').disabled)

    def test_multiple_uploads_questions_filter_and_reset(self):
        uploads = [pdf_upload('cv.pdf', 'Alice develops Python applications.'), pdf_upload('cv.pdf', 'Bob designs gardens and lawns.')]
        with patch('streamlit.file_uploader', return_value=uploads) as uploader, patch('resume_checker.qa.create_llm'), patch('resume_checker.qa.answer_question', return_value='Alice uses Python [Passage 1, page 1].') as answer:
            app = self.app().run()
            app.button(key='prepare').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.chat_input[0].disabled)
            app.chat_input[0].set_value('What skills are listed?').run()
            self.assertFalse(app.exception)
            passages = answer.call_args.args[2]
            self.assertEqual({p.metadata['document_id'] for p in passages}, {'1', '2'})
            self.assertEqual({p.metadata['source'] for p in passages}, {'cv.pdf'})
            self.assertEqual(len(app.session_state['messages']), 2)
            app.selectbox(key='cv_scope').select('CV 2: cv.pdf').run()
            app.chat_input[0].set_value('What does Bob do?').run()
            self.assertFalse(app.exception)
            self.assertEqual({p.metadata['document_id'] for p in answer.call_args.args[2]}, {'2'})
            answer.side_effect = RateLimitError(
                'sensitive error body',
                response=httpx.Response(429, request=httpx.Request('POST', 'https://api.groq.com')),
                body=None,
            )
            app.chat_input[0].set_value('Another question').run()
            self.assertFalse(app.exception)
            self.assertIn('429', app.error[0].value)
            self.assertNotIn('sensitive', app.error[0].value)
            self.assertFalse(app.chat_input[0].disabled)
            app.button(key='clear_chat').click().run()
            self.assertEqual(app.session_state['messages'], [])
            uploader.return_value = []
            app.run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state['cv_index'])
            self.assertTrue(app.chat_input[0].disabled)

    def test_invalid_pdf(self):
        with patch('streamlit.file_uploader', return_value=[SimpleNamespace(name='bad.pdf', getvalue=lambda: b'invalid')]):
            app = self.app().run()
            app.button(key='prepare').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertTrue(app.chat_input[0].disabled)


if __name__ == '__main__':
    unittest.main()
