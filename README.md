# MVB Compliance Review Prototype

A local Streamlit app for asking questions about compliance documents and running
selected checks from MVB's test workbook. The app has three tabs:

- **Conversation** answers questions about the indexed documents and displays
  source citations.
- **Testing** accepts PDF uploads, suggests matching test IDs, runs selected
  checks, and provides an Excel report.
- **Documents** lets you search the available corpus and inspect document
  metadata and indexed passages.

This is a proof of concept. Review every answer and test result against its
source documents; model output is not a compliance determination.

## Requirements

- Python 3.10 or later
- [Ollama](https://ollama.com/) running locally with the configured model
  available (default: `llama3.1:8b`)

## Set up and start the app

Run these commands from the project folder (the folder containing `app.py` and
`config.yaml`):

```powershell
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
python -m streamlit run app.py
```

Streamlit prints a local URL to open in your browser. If you use another Ollama
model, update `llm.model` in `config.yaml` or use the matching model there.

## Using the app

### Conversation

Ask a question in the chat box or choose one of the example prompts. Follow-up
questions use the conversation history as context. Answers show expandable
citations with document codes, page references, and supporting quotes. Citation
quotes are checked against retrieved text, but that does not verify the model's
interpretation.

### Testing

1. Upload one or more PDFs.
2. Review the suggested test IDs, choose a category, and select or change tests
   within that category. Selections are retained when you switch categories.
3. Choose **Run selected tests**.
4. Review the PASS, FAIL, and UNCLEAR results, then download the Excel report.

Suggestions are deterministic: the app matches document codes found in uploaded
PDFs to the configured workbook's file references, and recommends tests only
when their required documents are available. You can still select tests
manually.

Uploaded PDFs are combined with the configured corpus. If an upload has the
same document code as an indexed document, the upload replaces that document
for the current app session. Uploads are copied to a temporary directory on
the machine running Streamlit; the app does not provide a delete control.

### Documents

Search by document code, title, or filename. Choose a result to inspect its
metadata, page and passage counts, detected date, and extracted passages.

## Documents and configuration

The default configuration is in `config.yaml`:

- `chunks_file`: `data/chunks/chunks.jsonl` is the current indexed corpus.
  Set this to `null` to load PDFs from `documents_dir` instead.
- `tests_file`: `data/tests.xlsx`, the workbook used by the Testing tab and
  command-line runner.
- `llm`: model backend, name, local Ollama host, and request settings.
- `retrieval`: passage size and retrieval settings.
- `rules_file`: instructions used when evaluating workbook tests.
- `output_file` and `log_file`: destinations used by the command-line test
  runner.

The included chunk corpus contains section-start page metadata. Its page
references may not identify the exact page of every sentence; verify important
citations in the original PDFs. Scanned PDFs require OCR, which is not included.

## Command-line tools

The Streamlit app is the primary workflow. These commands are available for
testing or running the underlying pipeline from the project folder:

```powershell
# Run the unit tests
python -m unittest discover -s tests -v

# Check document and test-workbook mappings
python -m uc3.check

# Run a small plumbing check without calling a real model
python -m uc3.run --backend mock --limit 3

# Run selected tests with the configured model
python -m uc3.run --tests T36,T37,T38

# Ask questions in a terminal
python -m uc3.chat_cli
python -m uc3.chat_cli --question "How often are access reviews performed?"

# Score results against the answer key
python -m uc3.score
```

The mock backend checks application plumbing only; it does not assess factual
accuracy. The command-line test runner writes the Excel workbook to
`output/results.xlsx` and raw model responses to `output/run_log.jsonl` by
default. The Streamlit app instead offers its report as a download.

## Data handling

This prototype is intended for an approved local environment. Documents,
prompts, and model responses may contain sensitive information. The app sends
document excerpts to the configured model backend; with the default settings,
that backend is Ollama on the local machine. Do not upload confidential
documents unless the environment and model configuration are approved for them.

## Project layout

```text
app.py                  Streamlit app
config.yaml             Paths and model/retrieval settings
data/chunks/            Default indexed corpus (JSONL and CSV)
data/documents/         PDFs used when chunks_file is null
data/tests.xlsx         Test workbook
prompts/rules.txt       Test evaluation instructions
uc3/                    Document, retrieval, chat, test, and report pipeline
tests/                  Unit and integration tests
assets/mvb-logo.png     MVB logo used by the app
```
