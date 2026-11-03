"""Scoped retrieval: search only inside the document(s) a test names.

TF-IDF is used so the prototype needs no embedding model. To try embeddings later,
replace `rank_chunks` with a function that returns the same list of (score, Chunk).
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP = set("""a an the and or of to in on for with by is are was were be been that this these those it its as at from
any all each every must should shall will than then there their they them which who whom whose within without not no
provided document version client new more less greater least most under over per also into onto such""".split())


def keywords(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-/]{3,}|\d[\d,\.]*", text)
    return [w for w in words if w.lower() not in STOP]


def rank_chunks(query, chunks, top_k):
    if not chunks:
        return []
    texts = [f"{c.heading} {c.text}" for c in chunks]
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        m = vec.fit_transform(texts + [query])
        sims = cosine_similarity(m[-1], m[:-1]).ravel()
    except ValueError:
        return []
    ranked = sorted(zip(sims, chunks), key=lambda x: -x[0])
    return [(float(s), c) for s, c in ranked[:top_k] if s > 0]


class CorpusIndex:
    """Reusable TF-IDF index for repeated searches over an unchanged corpus."""

    def __init__(self, docs):
        self.chunks = [chunk for doc in docs for chunk in doc.chunks]
        self.vectorizer = None
        self.matrix = None
        if not self.chunks:
            return

        texts = [f"{chunk.heading} {chunk.text}" for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        try:
            self.matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            self.vectorizer = None

    def retrieve(self, query, top_k):
        """Return the top-k chunks across the corpus for a query."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query.strip() or self.matrix is None:
            return []
        try:
            query_vector = self.vectorizer.transform([query])
        except ValueError:
            return []
        if query_vector.nnz == 0:
            return []

        similarities = cosine_similarity(self.matrix, query_vector).ravel()
        ranked = sorted(
            zip(similarities, self.chunks),
            key=lambda item: -item[0],
        )
        return [
            (float(score), chunk)
            for score, chunk in ranked[:top_k]
            if score > 0
        ]


def retrieve(query, docs, top_k):
    """Top-k chunks PER document, so one document cannot crowd out another."""
    out = []
    for d in docs:
        hits = rank_chunks(query, d.chunks, top_k)
        if not hits and d.chunks:
            hits = [(0.0, d.chunks[0])]
        out += hits
    return out


def retrieve_corpus(query, docs, top_k):
    """Return the top-k chunks across the whole corpus for open-ended questions."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return CorpusIndex(docs).retrieve(query, top_k)
