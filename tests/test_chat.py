"""Tests for corpus-wide chat retrieval and citation verification."""
import json
import unittest

from uc3.chat import NO_RELEVANT_EVIDENCE, NO_VERIFIED_ANSWER, answer_question
from uc3.documents import Chunk, Document
from uc3.retrieve import CorpusIndex


class FakeBackend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        return self.response


def document(code, text, page=1):
    return Document(
        code=code,
        path=f"{code}.pdf",
        pages=[""] * (page - 1) + [text],
        chunks=[Chunk(code, page, page, "Policy", text)],
    )


class Chat(unittest.TestCase):
    def test_returns_citations_verified_against_retrieved_chunk(self):
        quote = "Access reviews are performed every quarter."
        docs = {
            "POL-001": document("POL-001", "Password changes require manager approval."),
            "POL-002": document("POL-002", quote, page=4),
        }
        backend = FakeBackend(json.dumps({
            "answer": "Access reviews are performed every quarter.",
            "citations": [{"document": "DOC POL-002", "page": 99, "quote": quote}],
        }))

        result = answer_question("How often are access reviews performed?", docs, backend, top_k=1)

        self.assertEqual(result.answer, "Access reviews are performed every quarter.")
        self.assertEqual(result.citations, [
            {"document": "POL-002", "page": 4, "quote": quote},
        ])
        self.assertEqual(len(backend.calls), 1)
        self.assertIn("DOC POL-002", backend.calls[0][1])
        self.assertNotIn("DOC POL-001", backend.calls[0][1])

    def test_rejects_unverified_citations_and_does_not_return_unsupported_answer(self):
        docs = {"POL-001": document("POL-001", "Access reviews are performed every quarter.")}
        backend = FakeBackend(json.dumps({
            "answer": "Access reviews happen every month.",
            "citations": [{"document": "POL-001", "page": 1, "quote": "Access reviews are performed every month."}],
        }))

        result = answer_question("How often are access reviews performed?", docs, backend)

        self.assertEqual(result.answer, NO_VERIFIED_ANSWER)
        self.assertEqual(result.citations, [])
        self.assertTrue(result.notes)

    def test_no_matching_chunks_skips_model_call(self):
        docs = {"POL-001": document("POL-001", "Access reviews are performed every quarter.")}
        backend = FakeBackend("{}")

        result = answer_question("What is the company's favorite color?", docs, backend)

        self.assertEqual(result.answer, NO_RELEVANT_EVIDENCE)
        self.assertEqual(backend.calls, [])

    def test_invalid_model_json_raises_clear_error(self):
        docs = {"POL-001": document("POL-001", "Access reviews are performed every quarter.")}
        backend = FakeBackend("not JSON")

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            answer_question("How often are access reviews performed?", docs, backend)

    def test_history_is_sent_and_used_for_retrieval_context(self):
        quote = "Privileged accounts require manager approval."
        docs = {"POL-002": document("POL-002", quote)}
        backend = FakeBackend(json.dumps({
            "answer": "Privileged accounts require manager approval.",
            "citations": [{"document": "POL-002", "quote": quote}],
        }))

        answer_question(
            "What about privileged ones?",
            docs,
            backend,
            history=[{"role": "user", "content": "What approval is needed for accounts?"}],
        )

        self.assertIn("CONVERSATION HISTORY", backend.calls[0][1])
        self.assertIn("CURRENT QUESTION:\nWhat about privileged ones?", backend.calls[0][1])

    def test_reuses_prebuilt_corpus_index(self):
        quote = "Access reviews are performed every quarter."
        docs = {"POL-001": document("POL-001", quote)}
        backend = FakeBackend(json.dumps({
            "answer": quote,
            "citations": [{"document": "POL-001", "quote": quote}],
        }))
        index = CorpusIndex(docs.values())

        result = answer_question(
            "How often are access reviews performed?",
            docs,
            backend,
            corpus_index=index,
        )

        self.assertEqual(result.citations[0]["quote"], quote)
        self.assertEqual(len(index.chunks), 1)

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            answer_question("  ", {}, FakeBackend("{}"))


if __name__ == "__main__":
    unittest.main()
