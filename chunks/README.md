# Step 1: Document Chunking

Part of the MVB RAG proof of concept (TCU BIS Capstone, MVB AI Scholars Program 2026, Use Case 3: Fintech Onboarding Document Q&A).

This step splits MVB's 30 synthetic onboarding documents into small, labeled chunks that a RAG pipeline can search.

## Files

- `chunk_docs.py`: the script that reads the PDFs and produces the chunks
- `chunks.jsonl`: the output, 270 chunks (9 per document), one per line. This is the input for the next step, embedding.

## How the documents are split

All 30 documents follow the same template: a header block, then numbered sections (1. Purpose, 2. Background, 3.1–3.4 Main Content, 4. Key Details table, 5. Exceptions, 6. Revision History). The script splits each document on these headings.

Design decisions:
- **Split by section, not by fixed size.** Sections are short (25–170 words) and each covers one topic, so they are natural units for search.
- **Not split by the 66 test types.** Grouping content by test type would require sorting the documents by hand, would create very large chunks, and would not work for new documents.
- **Header fields become metadata.** Doc ID, version, date, department, and related documents are attached to every chunk, so searches can be filtered by document and answers can cite their source.
- **Each chunk starts with its document title and section name**, so a passage like "$5,000 within 24 hours" still carries its context when searched on its own.
- **Tables are rewritten as one line per row** ("Label: value"), because raw PDF text scrambles table columns.

## Example chunk

```
{
  "chunk_id": "AML-001_3-2",
  "doc_id": "AML-001",
  "section": "3.2",
  "page": 1,
  "text": "AML Program Documentation (AML-001) | 3.2 Suspicious Activity Monitoring and SAR Thresholds\nPer this program's internal threshold policy, transactions or related transaction patterns aggregating $5,000 or more within a rolling 24-hour period are flagged for manual SAR evaluation..."
}
```

Only the `text` field is embedded. The other fields are stored as labels for filtering and citations.

## Run it

```
pip install pdfplumber
python chunk_docs.py <folder_with_pdfs> <output_folder>
```

The source PDFs are synthetic materials provided by MVB for this program and are not included in this repository.
