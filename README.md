## Teammate chunk integration (October 2026)

This updated package includes **270 section chunks from 30 documents** in
`data/chunks/chunks.jsonl`, with a review copy in `data/chunks/chunks.csv`.
The original PDF workflow is preserved. `config.yaml` now selects the JSONL
loader by default through `chunks_file`. Set `chunks_file: null` to revert
to reading PDFs from `data/documents/`.

From the folder containing `config.yaml`, run:

```cmd
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python -m uc3.check
python -m uc3.run --tests T36,T37,T38 --backend mock
```

The mock backend tests plumbing, **not factual AI accuracy**. To use the real
model, install/run Ollama and pull `llama3.1:8b`, then run
`python -m uc3.run --tests T36,T37,T38 --backend ollama`.

**Source fidelity:** JSONL metadata records each chunk's **starting page**, not
necessarily the exact page of every sentence. This loader reconstructs
page-indexed text from those chunks for compatibility with the existing
citation checker; verify important quotations against the original PDFs.
The existing tests of the original PDF loader still run independently.

**Security:** The supplied chunk data may contain internal onboarding
information. Do not commit this package to a **public** GitHub repository
without your team's authorization. Generated results and logs are ignored.

---

# UC3 prototype: RAG-assisted onboarding document review

Sample documents only. Reads MVB's test sheet (Excel), finds each test's document(s), gets a
PASS / FAIL / UNCLEAR verdict from a local model with a verified quote and page, and writes
a color-coded results Excel. **A human confirms every verdict.**

## Set up (about 15 minutes)
1. Python 3.10+ and a terminal in this folder.
2. `pip install -r requirements.txt`
3. Install Ollama (https://ollama.com), then pull a model that fits your laptop (check the Ollama library for current names):
   `ollama pull llama3.1:8b`   (7-8B models need roughly 16 GB RAM; smaller models are weaker at judging tests)
4. Set the same model name in `config.yaml` under `llm.model`.
5. Put the fintech's PDFs in `data/documents/` and MVB's test sheet at `data/tests.xlsx`.
   (A sample sheet and the ISO-001 PDF are already included.)

## Run it
```
python -m unittest                          # 15 checks that the pipeline logic works
python -m uc3.check                         # which tests have their documents? any code mismatches?
python -m uc3.run --backend mock            # plumbing test, no model needed (mock is NOT a real model)
python -m uc3.run --tests T36,T37,T38       # real model, a few tests first
python -m uc3.run                           # real model, every test
python -m uc3.score                         # compare to data/answer_key.csv
```
Results go to `output/results.xlsx` (Result, Evidence / Page Ref, Reason, Review Flag, Mode, Notes),
raw model replies to `output/run_log.jsonl`.

## How each test is handled (test by test, like MVB's analysts)
1. **Map:** the test's FILE NAME column names its document(s). Codes are matched hyphen-insensitively
   (SAN002 = SAN-002). We match on file name, not the code column, because T50-T53 list ISO-001 by mistake.
2. **Missing document -> UNCLEAR** (rule: a missing/wrong document is never a FAIL). No model call.
3. **"Created or reviewed within the last 12 months" -> answered in code** from the document's header date
   and revision history compared with `as_of_date`. No model call.
4. **Evidence:** if the test's documents fit under `full_doc_char_limit`, they are sent whole, with page markers.
   Otherwise the top-k chunks per document are retrieved (TF-IDF, scoped to that document, tables kept intact).
5. **Model:** `prompts/rules.txt` + the test + the evidence -> JSON verdict with citations.
6. **Safety checks (in code, not trusted to the model):**
   - every quote must exist in the evidence the model was given, otherwise the quote is dropped;
   - a PASS with no verified quote becomes UNCLEAR;
   - a retrieval-mode FAIL is downgraded to UNCLEAR if related text exists elsewhere in the document;
   - page numbers are corrected by locating the quote in the PDF;
   - invalid JSON becomes UNCLEAR, and one failed test never stops the run.

## What was verified, and what was not
- **Verified (here):** PDF parsing, header/date extraction, chunking with tables kept together, sheet parsing
  (including the three multi-document rows), scoped retrieval, both evidence modes, the date check, every safety
  check above, the Excel output, scoring, and the Ollama HTTP call against a fake local server.
- **NOT verified:** any real model's accuracy. No model ran in the build environment, so the first thing to do
  on your laptops is run 5-10 graded tests through Ollama and read `output/run_log.jsonl`.
- Only one real document (ISO-001) was available. Long documents, scanned PDFs, and messy tables are untested.

## Things to decide or tune
- `prompts/rules.txt`: items marked (CONFIRM) are team decisions (e.g. absence tests, conditional tests).
- Fill `data/answer_key.csv` (`Expected` column) as you grade tests by hand; keep some cases out of any prompt examples.
- Try two models and compare with `uc3.score`; change one thing at a time (model, `top_k`, `chunk_max_chars`).
- Scanned PDFs need OCR (not included). Tables are extracted as `| a | b |` text; check them per document.
- To try embeddings, replace `rank_chunks` in `uc3/retrieve.py` (same return shape).

## Layout
```
config.yaml            all settings
prompts/rules.txt      the PASS / FAIL / UNCLEAR rules sent to the model
uc3/sheet.py           read the test sheet, map tests to files
uc3/documents.py       PDF -> pages, header/date, structure-aware chunks
uc3/retrieve.py        scoped TF-IDF retrieval
uc3/dates.py           the 12-month check, in code
uc3/llm.py             Ollama backend + mock backend (one function: chat)
uc3/verdict.py         prompt building, model call, safety checks
uc3/run.py  check.py  score.py  report.py
tests/                 unit tests      tools/make_sample_tests.py  rebuilds the sample sheet from the PDF
```
