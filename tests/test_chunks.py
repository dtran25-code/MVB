"""Integration tests for teammate-supplied JSONL chunks."""
import unittest
from datetime import date
from uc3.documents import load_chunked_documents
from uc3.retrieve import retrieve

class ChunkIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_chunked_documents("data/chunks/chunks.jsonl")

    def test_document_and_chunk_counts(self):
        self.assertEqual(len(self.docs), 30)
        self.assertEqual(sum(len(d.chunks) for d in self.docs.values()), 270)

    def test_missing_codes_now_present(self):
        self.assertTrue({"AML-001", "CFT-001", "CDD-001", "TXM-001", "ONB-001"} <= self.docs.keys())

    def test_retrieval_returns_document_scoped_chunks(self):
        hits = retrieve("anti money laundering policy", [self.docs["AML-001"]], 3)
        self.assertTrue(hits)
        self.assertTrue(all(chunk.doc_code == "AML-001" for _, chunk in hits))

    def test_date_and_page_metadata(self):
        d = self.docs["AML-001"]
        self.assertIsInstance(d.doc_date, date)
        self.assertTrue(all(c.page_start >= 1 for c in d.chunks))

if __name__ == "__main__":
    unittest.main()
