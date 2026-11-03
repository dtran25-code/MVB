"""Run from the project folder:  python -m unittest -v"""
import json
import threading
import unittest
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml

from uc3.dates import months_between
from uc3.documents import load_documents
from uc3.llm import OllamaBackend
from uc3.retrieve import retrieve
from uc3.sheet import FILE_RE, Test, code_from_filename
from uc3.verdict import quote_in, norm, run_test

CFG = yaml.safe_load(open("config.yaml", encoding="utf8"))
DOCS = load_documents("data/documents")
RULES = "rules"
AS_OF = date(2026, 10, 14)


class FakeBackend:
    model = "fake"
    def __init__(self, payload): self.payload, self.calls = payload, 0
    def chat(self, system, user):
        self.calls += 1
        return json.dumps(self.payload)


class NoCallBackend:
    model = "nocall"
    def chat(self, *a): raise AssertionError("model must not be called")


def T(tid, text, code="ISO-001"):
    return Test(tid, text, [f"{code}_x.pdf"], [code], code, 0)


class SheetParsing(unittest.TestCase):
    def test_codes_hyphen_insensitive(self):
        self.assertEqual(code_from_filename("SAN002_Sanctions_Risk_Assessment.pdf"), "SAN-002")
        self.assertEqual(code_from_filename("ISP-001_Information_Security_Policy.pdf"), "ISP-001")

    def test_run_together_cell_is_split(self):
        cell = "SAN-001_Sanctions_Program.pdf, SAN002_Sanctions_Risk_Assessment.pdfSAN-001, SAN-002"
        self.assertEqual(FILE_RE.findall(cell), ["SAN-001_Sanctions_Program.pdf", "SAN002_Sanctions_Risk_Assessment.pdf"])


class Documents(unittest.TestCase):
    def test_header_date_and_pages(self):
        d = DOCS["ISO-001"]
        self.assertEqual(d.doc_date, date(2026, 3, 5))
        self.assertEqual(len(d.pages), 2)

    def test_table_stays_with_its_heading(self):
        d = DOCS["ISO-001"]
        key = next(c for c in d.chunks if c.heading.startswith("4. Key Details"))
        self.assertIn("| Major nonconformities | 0 |", key.text)

    def test_retrieval_finds_audit_findings(self):
        hits = retrieve("There were no major nonconformities identified during the audit.", [DOCS["ISO-001"]], 3)
        self.assertTrue(any("Audit Findings" in c.heading or "Key Details" in c.heading for _, c in hits))


class Checks(unittest.TestCase):
    def test_quote_check(self):
        hay = norm(DOCS["ISO-001"].full_text())
        self.assertTrue(quote_in("No major nonconformities were identified.", hay))
        self.assertTrue(quote_in("2 minor nonconformities ... No major nonconformities", hay))
        self.assertFalse(quote_in("All systems were found to be fully compliant.", hay))

    def test_month_math(self):
        self.assertEqual(months_between(date(2025, 10, 15), date(2026, 10, 14)), 11)
        self.assertEqual(months_between(date(2025, 10, 14), date(2026, 10, 14)), 12)


class RunTest(unittest.TestCase):
    def test_missing_document_is_unclear_and_skips_model(self):
        r = run_test(T("T50", "All production systems require MFA.", "ISP-001"), DOCS, NoCallBackend(), RULES, CFG, AS_OF)
        self.assertEqual((r.verdict, r.mode), ("UNCLEAR", "no-document"))

    def test_date_test_is_answered_in_code(self):
        r = run_test(T("T38", "The provided document version was created or reviewed within the last 12 months."),
                     DOCS, NoCallBackend(), RULES, CFG, AS_OF)
        self.assertEqual((r.verdict, r.mode), ("PASS", "date-check"))
        r = run_test(T("T38", "The provided document version was created or reviewed within the last 12 months."),
                     DOCS, NoCallBackend(), RULES, CFG, date(2027, 6, 1))
        self.assertEqual(r.verdict, "FAIL")

    def test_good_pass_keeps_verified_quote_and_real_page(self):
        b = FakeBackend({"verdict": "PASS", "reason": "stated", "review_flag": False, "answer": None,
                         "citations": [{"document": "ISO-001", "page": 9, "quote": "No major nonconformities were identified."}]})
        r = run_test(T("T37", "There were no major nonconformities identified during the audit."), DOCS, b, RULES, CFG, AS_OF)
        self.assertEqual(r.verdict, "PASS")
        self.assertEqual(r.citations[0]["page"], 1)          # corrected from the model's wrong page 9

    def test_fabricated_quote_downgrades_pass(self):
        b = FakeBackend({"verdict": "PASS", "reason": "x", "citations": [{"document": "ISO-001", "page": 1,
                         "quote": "The company has zero nonconformities of any kind."}]})
        r = run_test(T("T37", "There were no major nonconformities identified during the audit."), DOCS, b, RULES, CFG, AS_OF)
        self.assertEqual(r.verdict, "UNCLEAR")
        self.assertTrue(r.review_flag)

    def test_pass_without_quote_is_downgraded(self):
        b = FakeBackend({"verdict": "PASS", "reason": "x", "citations": []})
        r = run_test(T("T37", "no major nonconformities"), DOCS, b, RULES, CFG, AS_OF)
        self.assertEqual(r.verdict, "UNCLEAR")

    def test_bad_json_is_unclear(self):
        class Bad:
            model = "bad"
            def chat(self, *a): return "I think it passes!"
        r = run_test(T("T37", "no major nonconformities"), DOCS, Bad(), RULES, CFG, AS_OF)
        self.assertEqual(r.verdict, "UNCLEAR")

    def test_fail_not_mentioned_is_allowed_without_quote(self):
        b = FakeBackend({"verdict": "FAIL", "reason": "searched for OT/SCADA, nothing found", "citations": []})
        r = run_test(T("T52", "Policy addresses industrial control systems (OT/SCADA) security."), DOCS, b, RULES, CFG, AS_OF)
        self.assertEqual(r.verdict, "FAIL")


class OllamaPlumbing(unittest.TestCase):
    """Starts a tiny fake Ollama server and checks the real HTTP backend against it."""
    @classmethod
    def setUpClass(cls):
        class H(BaseHTTPRequestHandler):
            seen = {}
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                H.seen.update(body)
                reply = {"verdict": "PASS", "reason": "stated", "review_flag": False, "answer": None,
                         "citations": [{"document": "ISO-001", "page": 1, "quote": "No major nonconformities were identified."}]}
                out = json.dumps({"message": {"role": "assistant", "content": json.dumps(reply)}}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
            def log_message(self, *a): pass
        cls.H, cls.srv = H, HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()

    def test_end_to_end_through_http_backend(self):
        b = OllamaBackend("test-model", f"http://127.0.0.1:{self.srv.server_port}", 0, 4096, 10)
        r = run_test(T("T37", "There were no major nonconformities identified during the audit."), DOCS, b, "RULES", CFG, AS_OF)
        self.assertEqual(r.verdict, "PASS")
        self.assertEqual(self.H.seen["model"], "test-model")
        self.assertEqual(self.H.seen["format"], "json")
        self.assertEqual(self.H.seen["messages"][0]["content"], "RULES")
        self.assertIn("TEST: There were no major nonconformities", self.H.seen["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
