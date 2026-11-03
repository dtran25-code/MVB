"""'Created or reviewed within the last N months' is answered in code, not by the model."""
import re
from datetime import date

DATE_TEST = re.compile(r"created or reviewed within the last\s+(\d+)\s+months", re.I)


def is_date_test(text):
    m = DATE_TEST.search(text)
    return int(m.group(1)) if m else None


def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) - (1 if d2.day < d1.day else 0)


def check_document_date(doc, as_of, months):
    """Return (verdict, reason, evidence_text, page)."""
    if doc.doc_date is None:
        return "UNCLEAR", "No document date or revision history date could be found.", "", None
    age = months_between(doc.doc_date, as_of)
    src = doc.date_source
    page = 1 if "header" in src or "first page" in src else None
    if age <= months and doc.doc_date <= as_of:
        return ("PASS", f"Most recent date is {doc.doc_date.isoformat()} ({src}), {age} month(s) before "
                        f"{as_of.isoformat()}, within {months} months.", f"Date: {doc.doc_date.isoformat()}", page)
    if doc.doc_date > as_of:
        return ("UNCLEAR", f"Document date {doc.doc_date.isoformat()} is after the review date {as_of.isoformat()}.",
                f"Date: {doc.doc_date.isoformat()}", page)
    return ("FAIL", f"Most recent date is {doc.doc_date.isoformat()} ({src}), {age} months before "
                    f"{as_of.isoformat()}, older than {months} months.", f"Date: {doc.doc_date.isoformat()}", page)
