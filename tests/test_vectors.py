"""Embedding retrieval tests. A word-hashing stand-in replaces Ollama, so no model is needed."""
import hashlib
import math
import re
import tempfile
import unittest

import chromadb

from uc3.documents import Chunk, Document
from uc3.retrieve import make_index, retrieve
from uc3.vectors import DOC_PREFIX, QUERY_PREFIX, VectorIndex


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words get similar vectors."""
    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        self.inputs += texts
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in re.findall(r"[a-z0-9]+", t.lower().split(": ", 1)[-1]):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out


def doc(code, sections):
    d = Document(code=code, path=f"{code}.pdf", pages=["x"])
    d.chunks = [Chunk(code, 1, 1, h, t) for h, t in sections]
    return d


DOCS = [
    doc("FRD-002", [("3.2 Escalation", "High severity fraud cases are escalated within 48 hours."),
                    ("3.3 Reporting", "Monthly fraud reports go to the risk committee.")]),
    doc("ISP-001", [("3.1 Access", "Multi-factor authentication is required for all production systems."),
                    ("3.2 Encryption", "Customer data is encrypted at rest and in transit.")]),
]


class VectorIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = chromadb.PersistentClient(path=self.tmp.name)
        self.emb = FakeEmbedder()
        self.index = VectorIndex(DOCS, {}, embedder=self.emb, client=self.client)

    def tearDown(self):
        del self.index, self.client
        self.tmp.cleanup()

    def test_embeds_every_chunk_with_document_prefix(self):
        self.assertEqual((self.index.added, self.index.removed), (4, 0))
        self.assertTrue(all(t.startswith(DOC_PREFIX) for t in self.emb.inputs))

    def test_corpus_search_returns_scored_chunks(self):
        hits = self.index.retrieve("Is customer data encrypted in transit?", 2)
        self.assertEqual(hits[0][1].heading, "3.2 Encryption")
        self.assertGreater(hits[0][0], hits[1][0])
        self.assertTrue(self.emb.inputs[-1].startswith(QUERY_PREFIX))

    def test_document_filter(self):
        hits = self.index.search("fraud escalation", 5, ["ISP-001"])
        self.assertEqual({c.doc_code for _, c in hits}, {"ISP-001"})

    def test_per_document_retrieval_matches_pipeline_interface(self):
        hits = retrieve("high severity fraud escalated", DOCS, 1, index=self.index)
        self.assertEqual([c.doc_code for _, c in hits], ["FRD-002", "ISP-001"])
        self.assertEqual(hits[0][1].heading, "3.2 Escalation")

    def test_second_run_reuses_vectors_and_drops_stale_ones(self):
        again = VectorIndex(DOCS, {}, embedder=FakeEmbedder(), client=self.client)
        self.assertEqual((again.added, again.removed), (0, 0))
        changed = VectorIndex(DOCS[:1], {}, embedder=FakeEmbedder(), client=self.client)
        self.assertEqual((changed.added, changed.removed), (0, 2))
        self.assertEqual(changed.collection.count(), 2)


class MakeIndexTest(unittest.TestCase):
    def test_default_is_tfidf(self):
        self.assertFalse(hasattr(make_index(DOCS, {}), "retrieve_per_document"))

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            make_index(DOCS, {"retrieval": {"method": "bm25"}})


if __name__ == "__main__":
    unittest.main()
