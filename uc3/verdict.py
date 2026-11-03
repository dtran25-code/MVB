"""Turn one test into one verdict: evidence -> model -> safety checks."""
import json
import re
from dataclasses import dataclass, field

from .dates import check_document_date, is_date_test
from .retrieve import keywords, retrieve

VALID = {"PASS", "FAIL", "UNCLEAR"}


@dataclass
class Result:
    test_id: str
    verdict: str
    reason: str
    citations: list = field(default_factory=list)   # [{"document","page","quote"}]
    answer: str = None
    review_flag: bool = True
    mode: str = ""                                  # date-check | full-document | retrieval | no-document
    documents: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    raw: str = ""


def norm(s):
    s = (s or "").lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"[^a-z0-9$%.]+", " ", s).strip()


def quote_in(quote, haystack_norm):
    """Every non-trivial fragment of the quote (split on ...) must appear in the evidence."""
    frags = [norm(f) for f in re.split(r"\.\.\.|\u2026", quote or "")]
    frags = [f for f in frags if len(f) >= 8]
    return bool(frags) and all(f in haystack_norm for f in frags)


def find_page(quote, doc):
    q = norm(re.split(r"\.\.\.|\u2026", quote or "")[0])
    for i, p in enumerate(doc.pages, start=1):
        if q and q in norm(p):
            return i
    return None


def build_prompt(test, docs, cfg_ret):
    total = sum(d.chars for d in docs)
    if total <= cfg_ret["full_doc_char_limit"]:
        mode = "full-document"
        parts = [f"[DOC {d.code}]\n{d.full_text()}" for d in docs]
        shown = {d.code: d.full_text() for d in docs}
    else:
        mode = "retrieval"
        hits = retrieve(test.text, docs, cfg_ret["top_k"])
        parts, shown = [], {d.code: "" for d in docs}
        for score, c in hits:
            pg = f"{c.page_start}" if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
            parts.append(f"[DOC {c.doc_code} | PAGE {pg} | SECTION {c.heading}]\n{c.text}")
            shown[c.doc_code] += "\n" + c.text
    user = (f"TEST ID: {test.test_id}\nTEST: {test.text}\n"
            f"DOCUMENTS NAMED FOR THIS TEST: {', '.join(d.code for d in docs)}\n"
            f"EVIDENCE MODE: {mode}\n\nEVIDENCE START\n" + "\n\n".join(parts) + "\n\nEVIDENCE END\n"
            "Respond with the JSON object only.")
    return mode, user, shown


def parse_json(raw):
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def run_test(test, documents, backend, rules, cfg, as_of, log=None):
    docs, missing = [], []
    for code in test.codes:
        (docs if code in documents else missing).append(documents[code] if code in documents else code)

    # Rule 18: wrong/missing document is UNCLEAR, never FAIL. Decided in code, no model call.
    if not test.codes or missing:
        why = "no document is named for this test" if not test.codes else \
              f"document(s) not found in the documents folder: {', '.join(missing)}"
        r = Result(test.test_id, "UNCLEAR", f"Cannot evaluate: {why}.", mode="no-document", review_flag=True)
        if test.declared_codes and test.codes and test.declared_codes.replace(" ", "") != ",".join(test.codes):
            r.notes.append(f"Sheet lists code(s) '{test.declared_codes}' but file name(s) imply {test.codes}")
        return r

    # Date tests are answered in code.
    months = is_date_test(test.text)
    if months:
        verdicts = [check_document_date(d, as_of, months) for d in docs]
        worst = "FAIL" if any(v[0] == "FAIL" for v in verdicts) else \
                "UNCLEAR" if any(v[0] == "UNCLEAR" for v in verdicts) else "PASS"
        cites = [{"document": d.code, "page": v[3], "quote": v[2]} for d, v in zip(docs, verdicts) if v[2]]
        return Result(test.test_id, worst, " ".join(v[1] for v in verdicts), cites, mode="date-check",
                      documents=[d.code for d in docs], review_flag=(worst != "PASS"))

    mode, user, shown = build_prompt(test, docs, cfg["retrieval"])
    raw = backend.chat(rules, user)
    if log is not None:
        log.write(json.dumps({"test": test.test_id, "mode": mode, "prompt_chars": len(user), "response": raw}) + "\n")
    res = Result(test.test_id, "UNCLEAR", "", mode=mode, documents=[d.code for d in docs], raw=raw)
    data = parse_json(raw)
    if not data or str(data.get("verdict", "")).upper() not in VALID:
        res.reason = "Model did not return valid JSON with a verdict; needs manual review."
        res.notes.append("invalid model output")
        return res
    res.verdict = str(data["verdict"]).upper()
    res.reason = str(data.get("reason", "")).strip()
    res.answer = data.get("answer")
    res.review_flag = bool(data.get("review_flag", False))
    cites = [c for c in (data.get("citations") or []) if isinstance(c, dict)]

    # Safety check 1: every quote must exist in the evidence the model was given.
    hay = norm(" ".join(shown.values()))
    bad = [c for c in cites if not quote_in(c.get("quote", ""), hay)]
    good = [c for c in cites if c not in bad]
    for c in good:
        owner = next((d for d in docs if d.code == c.get("document")), None) or \
                next((d for d in docs if find_page(c.get("quote", ""), d)), None)
        if owner:
            c["document"] = owner.code
            c["page"] = find_page(c.get("quote", ""), owner) or c.get("page")
    res.citations = good
    if bad:
        res.notes.append(f"{len(bad)} quote(s) not found in the source text and were removed")
    if res.verdict == "PASS" and (not good or bad):
        res.verdict, res.review_flag = "UNCLEAR", True
        res.notes.append("PASS downgraded: supporting quote could not be verified")
    if res.verdict == "FAIL" and bad:
        res.verdict, res.review_flag = "UNCLEAR", True
        res.notes.append("FAIL downgraded: cited contradiction could not be verified")

    # Safety check 2: a FAIL for 'not mentioned' must not be a retrieval miss.
    if res.verdict == "FAIL" and mode == "retrieval" and not good:
        kws = {k.lower() for k in keywords(test.text)}
        hits = []
        for d in docs:
            for c in d.chunks:
                if c.text not in shown[d.code]:
                    found = {k for k in kws if k in c.text.lower()}
                    if len(found) >= max(2, len(kws) // 2):
                        hits.append(f"{d.code} p.{c.page_start} ({', '.join(sorted(found))})")
        if hits:
            res.verdict, res.review_flag = "UNCLEAR", True
            res.notes.append("FAIL downgraded: related text exists outside retrieved passages: " + "; ".join(hits[:3]))

    if res.verdict != "PASS":
        res.review_flag = True
    return res
