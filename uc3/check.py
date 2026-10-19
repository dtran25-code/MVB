"""Diagnostics BEFORE running a model:   python -m uc3.check
Shows which tests have their documents, what dates were found, and how each document would be fed to the model."""
import sys
import yaml
from .documents import load_documents, load_chunked_documents
from .sheet import load_tests


def main():
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml", encoding="utf8"))
    docs = (load_chunked_documents(cfg["chunks_file"]) if cfg.get("chunks_file")
            else load_documents(cfg["documents_dir"], cfg["retrieval"]["chunk_max_chars"]))
    tests = load_tests(cfg["tests_file"])
    limit = cfg["retrieval"]["full_doc_char_limit"]
    print("DOCUMENTS FOUND")
    for c, d in sorted(docs.items()):
        mode = "full-document" if d.chars <= limit else "retrieval"
        print(f"  {c}: {len(d.pages)} pages, {d.chars:,} chars, {len(d.chunks)} chunks, date={d.doc_date} ({d.date_source or 'none'}), mode={mode}")
    print("\nTESTS")
    missing = {}
    for t in tests:
        gone = [c for c in t.codes if c not in docs]
        for c in gone:
            missing.setdefault(c, []).append(t.test_id)
        note = ""
        if t.declared_codes and t.codes and t.declared_codes.replace(" ", "") != ",".join(t.codes):
            note = f"   <-- sheet code '{t.declared_codes}' differs from file name code {t.codes}"
        if not t.codes:
            note = "   <-- no file name found for this test"
        if note:
            print(f"  {t.test_id}{note}")
    print(f"\n{len(tests)} tests; {sum(1 for t in tests if t.codes and all(c in docs for c in t.codes))} have all documents available")
    if missing:
        print("\nMISSING DOCUMENTS (tests that cannot run yet)")
        for c, ids in sorted(missing.items()):
            print(f"  {c}: {', '.join(ids)}")


if __name__ == "__main__":
    main()
