"""Load PDFs, keep page numbers, keep tables intact, and build structure-aware chunks."""
import re
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import pdfplumber

from .sheet import code_from_filename

HEADING_RE = re.compile(r"^(\d+(?:\.\d+)+|\d+\.)\s+([A-Z][^\n]{1,90})$")
ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
LONG_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(20\d{2})\b")
MONTHS = {m: i + 1 for i, m in enumerate(
    "January February March April May June July August September October November December".split())}
HEADER_LABELS = {"doc id", "title", "organization", "doc type", "date", "version", "author",
                 "department", "status", "related docs"}


@dataclass
class Chunk:
    doc_code: str
    page_start: int
    page_end: int
    heading: str
    text: str


@dataclass
class Document:
    code: str
    path: str
    pages: list                       # list[str], index 0 = page 1
    header: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    doc_date: date = None
    date_source: str = ""

    @property
    def chars(self):
        return sum(len(p) for p in self.pages)

    def full_text(self):
        return "\n".join(f"[PAGE {i + 1}]\n{p}" for i, p in enumerate(self.pages))


def _table_md(rows):
    out = []
    for r in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def extract_page_elements(page):
    """Return [(kind, text)] in reading order; tables are kept as one 'table' element."""
    tables = page.find_tables()
    boxes = [t.bbox for t in tables]
    elems = []
    for ln in page.extract_text_lines(layout=False):
        mid = (ln["top"] + ln["bottom"]) / 2
        if any(b[1] - 1 <= mid <= b[3] + 1 and b[0] - 2 <= ln["x0"] <= b[2] + 2 for b in boxes):
            continue
        elems.append((ln["top"], "line", ln["text"].strip()))
    for t in tables:
        rows = t.extract()
        if rows:
            elems.append((t.bbox[1], "table", _table_md(rows)))
    elems.sort(key=lambda e: e[0])
    return [(k, txt) for _, k, txt in elems if txt]


def parse_header(first_page_lines):
    """Header fields appear as 'Label: value' OR as a block of labels followed by a block of values."""
    hdr = {}
    lines = [l for l in first_page_lines[:40]]
    labels = []
    for l in lines:
        key = l.rstrip(":").strip().lower()
        if l.endswith(":") and key in HEADER_LABELS:
            labels.append(key)
        elif labels:
            break
    if labels:
        start = len(labels)
        values = lines[start:start + len(labels)]
        hdr.update(dict(zip(labels, values)))
    for l in lines:
        m = re.match(r"^([A-Za-z ]+):\s+(.+)$", l)
        if m and m.group(1).strip().lower() in HEADER_LABELS:
            hdr[m.group(1).strip().lower()] = m.group(2).strip()
    return hdr


def _dates_in(text):
    found = []
    for y, m, d in ISO_DATE.findall(text):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    for mon, d, y in LONG_DATE.findall(text):
        try:
            found.append(date(int(y), MONTHS[mon], int(d)))
        except ValueError:
            pass
    return found


def find_document_date(doc):
    """Latest date among the header 'Date' field and the revision-history section."""
    cands = []
    if "date" in doc.header:
        cands += [(d, "header") for d in _dates_in(doc.header["date"])]
    for ch in doc.chunks:
        if re.search(r"revision history|version history|document control", ch.heading, re.I):
            cands += [(d, f"revision history p.{ch.page_start}") for d in _dates_in(ch.text)]
    if not cands:
        cands = [(d, "first page") for d in _dates_in(doc.pages[0] if doc.pages else "")]
    if not cands:
        return None, ""
    d, src = max(cands, key=lambda x: x[0])
    return d, src


def build_chunks(code, elements, max_chars):
    """elements: [(page, kind, text)]. One chunk per heading; tables are never split."""
    chunks, cur = [], None
    path = []

    def flush():
        nonlocal cur
        if cur and cur["units"]:
            label = " > ".join(cur["path"]) or "HEADER"
            units, buf, size, p0 = cur["units"], [], 0, None
            for pg, txt in units:
                if buf and size + len(txt) > max_chars:
                    chunks.append(Chunk(code, p0, buf_pages[-1], label, "\n".join(buf)))
                    buf, size, p0 = [], 0, None
                if p0 is None:
                    p0, buf_pages = pg, []
                buf.append(txt); buf_pages.append(pg); size += len(txt)
            if buf:
                chunks.append(Chunk(code, p0, buf_pages[-1], label, "\n".join(buf)))
        cur = None

    cur = {"path": [], "units": []}
    for pg, kind, txt in elements:
        m = HEADING_RE.match(txt) if kind == "line" else None
        if m and not txt.endswith("."):
            flush()
            depth = txt.split()[0].rstrip(".").count(".") + 1
            path = path[:depth - 1] + [txt]
            cur = {"path": list(path), "units": [(pg, txt)]}
        else:
            if cur is None:
                cur = {"path": list(path), "units": []}
            cur["units"].append((pg, txt))
    flush()
    return chunks


def load_document(path, max_chars=1500):
    path = Path(path)
    code = code_from_filename(path.name) or path.stem
    pages, elements = [], []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            elems = extract_page_elements(page)
            pages.append("\n".join(t for _, t in elems))
            elements += [(i, k, t) for k, t in elems]
    doc = Document(code=code, path=str(path), pages=pages)
    first_lines = [t for p, k, t in elements if p == 1 and k == "line"]
    doc.header = parse_header(first_lines)
    doc.chunks = build_chunks(code, elements, max_chars)
    doc.doc_date, doc.date_source = find_document_date(doc)
    return doc


def load_documents(folder, max_chars=1500):
    docs = {}
    for p in sorted(Path(folder).glob("*.pdf")):
        d = load_document(p, max_chars)
        if d.code in docs:
            print(f"WARNING: two files map to code {d.code}: {docs[d.code].path} and {p}")
        docs[d.code] = d
    return docs


def load_chunked_documents(jsonl_path):
    """Load teammate-provided section chunks without requiring the source PDFs.

    Page numbers are the section-start pages supplied by the chunker; when a
    chunk spans pages, citations cannot be more precise than that metadata.
    """
    path = Path(jsonl_path)
    docs = {}
    page_parts = {}
    with path.open(encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                code = str(item["doc_id"]).strip().upper()
                body = str(item["text"]).strip()
                page = int(item["page"])
                if not code or not body or page < 1:
                    raise ValueError("empty document id/text or invalid page")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid chunk in {path} line {line_no}: {exc}") from exc
            if code not in docs:
                hdr = {k: str(item.get(k, "") or "") for k in
                       ("version", "date", "author", "department", "status")}
                hdr["title"] = str(item.get("doc_title", ""))
                d = Document(code=code, path=str(item.get("source_file") or path), pages=[], header=hdr)
                docs[code] = d
                page_parts[code] = {}
            d = docs[code]
            heading = f"{item.get('section', '')} {item.get('section_title', '')}".strip()
            d.chunks.append(Chunk(code, page, page, heading, body))
            page_parts[code].setdefault(page, []).append(body)
    for code, d in docs.items():
        sections = page_parts[code]
        d.pages = ["\n".join(sections.get(pg, [])) for pg in range(1, max(sections) + 1)]
        # These are reconstructed page-indexed chunk texts, not the original PDF pages.
        d.doc_date, d.date_source = find_document_date(d)
    return docs
