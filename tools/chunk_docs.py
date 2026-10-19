"""
chunk_docs.py - split the Veridian Dynamics onboarding documents into RAG chunks.

Usage:
    python chunk_docs.py <folder_with_pdfs> <output_folder>

What it does, for every PDF in the folder:
  1. Reads the header block (Doc Id, Title, Version, Date, ...) and keeps it as metadata.
  2. Splits the body on the numbered section headings (1., 2., 3.1-3.x, 4., 5., 6.).
  3. Rebuilds the tables (Key Details, Revision History) as one "Label: value" line per row,
     because raw PDF text scrambles table columns.
  4. Puts "Document title (DOC-ID) | Section" at the top of every chunk so search knows
     where the text came from.

Outputs:
  <output_folder>/chunks.jsonl   one chunk per line, for the pipeline (vector DB loading)
  <output_folder>/chunks.csv     same chunks, for the team to review in Excel
  <output_folder>/md/<DOC-ID>.md one cleaned Markdown file per document

Requires: pip install pdfplumber
"""

import csv
import json
import re
import sys
from pathlib import Path

import pdfplumber

HEADER_FIELDS = ["Doc Id", "Title", "Organization", "Doc Type", "Date", "Version",
                 "Author", "Department", "Status", "Related Docs"]

# Top-level headings used by every document, plus 3.x subsections.
TOP_HEADING = re.compile(
    r"^([1-6])\.\s+(Purpose/Summary|Background/Context|Main Content|Key Details|"
    r"Exceptions, Risks, or Open Issues|References and Revision History)\s*$")
SUB_HEADING = re.compile(r"^(3\.\d)\s+(.+?)\s*$")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

SECTION_TYPE = {
    "1": "purpose", "2": "background", "3": "main_content",
    "4": "key_details", "5": "exceptions", "6": "revision_history",
}


def parse_header(lines):
    """Read 'Label: value' lines at the top of page 1. Returns (metadata, index of first body line)."""
    meta, i = {}, 0
    while i < len(lines) and len(meta) < len(HEADER_FIELDS):
        m = re.match(r"^(%s):\s*(.*)$" % "|".join(re.escape(f) for f in HEADER_FIELDS), lines[i])
        if not m:
            break
        meta[m.group(1)] = m.group(2).strip()
        i += 1

    # Some PDFs wrap the Doc Type onto the Date line ("Date: Documentation 2026-02-28").
    date_line = meta.get("Date", "")
    d = DATE_RE.search(date_line)
    if d:
        spill = date_line.replace(d.group(0), "").strip()
        if spill:
            meta["Doc Type"] = (meta.get("Doc Type", "") + " " + spill).strip()
        meta["Date"] = d.group(0)

    # Same wrapping problem on the Organization line ("Veridian Dynamics ACH" + "Compliance Policy").
    org = meta.get("Organization", "")
    if org.startswith("Veridian Dynamics") and org != "Veridian Dynamics":
        spill = org[len("Veridian Dynamics"):].strip()
        meta["Organization"] = "Veridian Dynamics"
        meta["Doc Type"] = (spill + " " + meta.get("Doc Type", "")).strip()

    related = meta.get("Related Docs", "").strip("[] ")
    meta["Related Docs"] = [r.strip() for r in related.split(",") if r.strip()]

    # The document title is repeated once right after the header; skip it.
    if i < len(lines) and lines[i].strip() == meta.get("Title", "").strip():
        i += 1
    return meta, i


def table_to_lines(table):
    """Turn a table (list of rows) into readable 'Label: value' lines."""
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in table if r]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    out = []
    for r in body:
        if len(r) == 2:
            out.append(f"{r[0]}: {r[1]}")
        else:
            parts = [f"{header[j]}: {r[j]}" for j in range(1, len(r)) if j < len(header)]
            out.append(f"{header[0]}: {r[0]} | " + "; ".join(parts))
    return out


def extract(pdf_path):
    """Return metadata, body lines tagged with page numbers, and tables in reading order."""
    pages_lines, tables = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            found = page.find_tables()
            boxes = [t.bbox for t in found]

            def outside(obj, boxes=boxes):
                cx = (obj["x0"] + obj["x1"]) / 2
                cy = (obj["top"] + obj["bottom"]) / 2
                return not any(x0 <= cx <= x1 and t <= cy <= b for x0, t, x1, b in boxes)

            text = page.filter(outside).extract_text() or ""
            lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
            pages_lines.append((pno, lines))
            for t in found:
                tables.append({"page": pno, "rows": t.extract()})

    first_page_lines = pages_lines[0][1]
    meta, start = parse_header(first_page_lines)
    pages_lines[0] = (pages_lines[0][0], first_page_lines[start:])

    body = [(pno, ln) for pno, lines in pages_lines for ln in lines]
    return meta, body, tables


def split_sections(body):
    """Group body lines under their section heading."""
    sections, cur = [], None
    for pno, ln in body:
        top, sub = TOP_HEADING.match(ln), SUB_HEADING.match(ln)
        if top:
            cur = {"num": top.group(1), "title": top.group(2), "page": pno, "lines": []}
            sections.append(cur)
        elif sub and cur is not None and cur["num"].startswith("3"):
            cur = {"num": sub.group(1), "title": sub.group(2), "page": pno, "lines": []}
            sections.append(cur)
        elif cur is not None:
            cur["lines"].append(ln)
    return sections


def merge_tables(tables):
    """Classify tables as key details or revision history; join tables split across pages."""
    key, rev = [], []
    for t in tables:
        rows = t["rows"]
        if not rows:
            continue
        first = [(c or "").strip() for c in rows[0]]
        if first and first[0] == "Version" and "Change Summary" in first:
            rev = rows
        elif rev:            # continuation of the revision table on the next page
            rev += rows
        elif key and first[0] not in ("Element", "Metric", "Tier", "Term", "Coverage Type", "Risk Factor"):
            key += rows      # continuation of the key details table on the next page
        else:
            key = rows
    return key, rev


def build_chunks(pdf_path):
    meta, body, tables = extract(pdf_path)
    sections = split_sections(body)
    key_table, rev_table = merge_tables(tables)

    # The tags line sits at the very end, sometimes glued to the last table row.
    tags = []
    for s in sections:
        kept = []
        for ln in s["lines"]:
            m = re.search(r"Tags:\s*(.*)$", ln)
            if m:
                tags = [t.strip() for t in m.group(1).split(",") if t.strip()]
                before = ln[:m.start()].strip()
                if before:
                    kept.append(before)
            else:
                kept.append(ln)
        s["lines"] = kept

    doc_id, title = meta.get("Doc Id", Path(pdf_path).stem), meta.get("Title", "")
    chunks, md_parts = [], [f"# {title}", ""]
    for k in HEADER_FIELDS:
        v = meta.get(k, "")
        md_parts.append(f"- **{k}:** {', '.join(v) if isinstance(v, list) else v}")
    md_parts.append("")

    for s in sections:
        num = s["num"]
        top = num.split(".")[0]
        if num == "3":                      # "3. Main Content" is only a parent heading
            md_parts += [f"## 3. {s['title']}", ""]
            continue
        if top == "4" and key_table:
            text_lines = table_to_lines(key_table)
        elif top == "6" and rev_table:
            text_lines = table_to_lines(rev_table)
        else:
            text_lines = s["lines"]
        body_text = " ".join(text_lines) if top not in ("4", "6") else "\n".join(text_lines)
        body_text = re.sub(r"\s+-\s*$", "", body_text).strip()
        if not body_text:
            continue

        heading = f"{num} {s['title']}" if "." in num else f"{num}. {s['title']}"
        md_parts += [f"{'###' if '.' in num else '##'} {heading}", "",
                     body_text if top not in ("4", "6") else "\n".join(f"- {l}" for l in text_lines), ""]

        chunks.append({
            "chunk_id": f"{doc_id}_{num.replace('.', '-')}",
            "doc_id": doc_id,
            "doc_title": title,
            "section": num,
            "section_title": s["title"],
            "section_type": SECTION_TYPE.get(top, "other"),
            "page": s["page"],
            "version": meta.get("Version", ""),
            "date": meta.get("Date", ""),
            "author": meta.get("Author", ""),
            "department": meta.get("Department", ""),
            "status": meta.get("Status", ""),
            "related_docs": meta.get("Related Docs", []),
            "tags": tags,
            "source_file": Path(pdf_path).name,
            # The text that gets embedded and shown to the model:
            "text": f"{title} ({doc_id}) | {heading}\n{body_text}",
        })

    if tags:
        md_parts += [f"*Tags: {', '.join(tags)}*", ""]
    return chunks, "\n".join(md_parts)


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    (out_dir / "md").mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for pdf in sorted(in_dir.glob("*.pdf")):
        chunks, md = build_chunks(pdf)
        doc_id = chunks[0]["doc_id"] if chunks else pdf.stem
        (out_dir / "md" / f"{doc_id}.md").write_text(md, encoding="utf-8")
        all_chunks += chunks
        print(f"{doc_id:10s} {len(chunks):3d} chunks  ({pdf.name})")

    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    cols = ["chunk_id", "doc_id", "doc_title", "section", "section_title", "section_type",
            "page", "version", "date", "department", "related_docs", "tags", "text"]
    with open(out_dir / "chunks.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in all_chunks:
            row = dict(c)
            row["related_docs"] = ", ".join(c["related_docs"])
            row["tags"] = ", ".join(c["tags"])
            w.writerow(row)

    print(f"\nTotal: {len(all_chunks)} chunks from {len(list(in_dir.glob('*.pdf')))} documents")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python chunk_docs.py <folder_with_pdfs> <output_folder>")
    main(sys.argv[1], sys.argv[2])
