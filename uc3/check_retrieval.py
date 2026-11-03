"""Retrieval check: does search find the right document for each test? No LLM needed.

    python -m uc3.check_retrieval                 # compare TF-IDF and embedding search
    python -m uc3.check_retrieval --method embedding

For every test in the workbook, the test text is searched across ALL chunks (no
document filter) and we record where the first chunk from the test's own
document lands. A higher top-k hit rate means search understands the test better.
The test runner itself filters to the named document, so this is a harder check.
Per-test results are written to output/retrieval_check.csv.
"""
import argparse
import csv
from pathlib import Path

import yaml

from .documents import load_chunked_documents, load_documents
from .retrieve import make_index
from .sheet import load_tests

CUTOFFS = (1, 3, 5)


def first_hit_rank(hits, expected):
    return next((i + 1 for i, (_, c) in enumerate(hits) if c.doc_code in expected), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check search quality against the test workbook")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method", choices=["tfidf", "embedding", "both"], default="both")
    ap.add_argument("--output", default="output/retrieval_check.csv")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf8"))
    docs = (load_chunked_documents(cfg["chunks_file"]) if cfg.get("chunks_file")
            else load_documents(cfg["documents_dir"], cfg["retrieval"]["chunk_max_chars"]))
    tests = [t for t in load_tests(cfg["tests_file"]) if t.codes]
    methods = ["tfidf", "embedding"] if a.method == "both" else [a.method]
    top = max(CUTOFFS)

    rows = {t.test_id: {"test": t.test_id, "expected_docs": ", ".join(t.codes), "test_text": t.text}
            for t in tests}
    print(f"{len(tests)} tests, {sum(len(d.chunks) for d in docs.values())} chunks\n")
    for method in methods:
        index = make_index(docs.values(), {**cfg, "retrieval": {**cfg["retrieval"], "method": method}})
        hits_at = {k: 0 for k in CUTOFFS}
        for t in tests:
            hits = index.retrieve(t.text, top)
            rank = first_hit_rank(hits, set(t.codes))
            for k in CUTOFFS:
                hits_at[k] += bool(rank and rank <= k)
            rows[t.test_id][f"{method}_rank"] = rank or "miss"
            rows[t.test_id][f"{method}_top{top}"] = "; ".join(
                f"{c.doc_code} {c.heading[:40]} ({s:.2f})" for s, c in hits)
        print(f"{method:>9}: " + "   ".join(
            f"top {k}: {hits_at[k]}/{len(tests)} = {hits_at[k] / len(tests):.0%}" for k in CUTOFFS))

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fields = list(next(iter(rows.values())).keys())
    with open(a.output, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows.values())
    print(f"\nPer-test results: {a.output}")


if __name__ == "__main__":
    main()
