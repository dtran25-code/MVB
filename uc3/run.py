"""Run the test sheet through the pipeline.   python -m uc3.run --help"""
import argparse
from datetime import date, datetime
from pathlib import Path
import yaml

from .documents import load_documents, load_chunked_documents
from .llm import make_backend
from .report import write_results
from .sheet import load_sheet, load_tests
from .verdict import run_test


def load_rules(path):
    return "\n".join(l for l in Path(path).read_text(encoding="utf8").splitlines() if not l.lstrip().startswith("#"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run UC3 tests through the pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tests", help="comma-separated test ids to run, e.g. T36,T37")
    ap.add_argument("--limit", type=int, help="run only the first N matching tests")
    ap.add_argument("--backend", choices=["ollama", "mock"], help="override config backend")
    ap.add_argument("--model", help="override config model name")
    ap.add_argument("--as-of", help="YYYY-MM-DD, date used for the 12-month check")
    ap.add_argument("--output", help="override output file")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf8"))
    if a.backend: cfg["llm"]["backend"] = a.backend
    if a.model: cfg["llm"]["model"] = a.model
    as_of = date.fromisoformat(a.as_of or cfg.get("as_of_date") or date.today().isoformat())
    out = a.output or cfg["output_file"]

    print(f"Loading documents from {cfg['documents_dir']} ...")
    docs = (load_chunked_documents(cfg["chunks_file"]) if cfg.get("chunks_file")
            else load_documents(cfg["documents_dir"], cfg["retrieval"]["chunk_max_chars"]))
    print(f"  {len(docs)} document(s): {', '.join(sorted(docs)) or 'none'}")
    tests = load_tests(cfg["tests_file"])
    if a.tests:
        want = {t.strip().upper() for t in a.tests.split(",")}
        tests = [t for t in tests if t.test_id.upper() in want]
    if a.limit:
        tests = tests[:a.limit]
    backend = make_backend(cfg["llm"])
    model_name = getattr(backend, "model", cfg["llm"]["model"])
    rules = load_rules(cfg["rules_file"])
    print(f"Running {len(tests)} test(s) with {cfg['llm']['backend']}:{model_name}  (as-of {as_of})\n")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(cfg["log_file"], "w", encoding="utf8") as log:
        for t in tests:
            start = datetime.now()
            try:
                r = run_test(t, docs, backend, rules, cfg, as_of, log)
            except Exception as e:  # one bad test must not stop the run
                from .verdict import Result
                r = Result(t.test_id, "UNCLEAR", f"Error while running this test: {e}", mode="error")
            secs = (datetime.now() - start).total_seconds()
            results.append(r)
            flag = " [review]" if r.review_flag else ""
            print(f"{t.test_id:>4}  {r.verdict:<7} {r.mode:<14} {secs:5.1f}s{flag}  {r.reason[:80]}")
    write_results(load_sheet(cfg["tests_file"]), tests, results, out, model_name, as_of, cfg)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
