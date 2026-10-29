"""Streamlit interface for document chat and selected test runs."""
from datetime import date
import hashlib
from pathlib import Path
import tempfile
import time

import requests
import streamlit as st
import yaml
from pdfminer.pdfparser import PDFSyntaxError

from uc3.chat import answer_question
from uc3.documents import load_chunked_documents, load_document, load_documents
from uc3.llm import make_backend
from uc3.report import write_results
from uc3.retrieve import CorpusIndex
from uc3.run import evaluate_tests, load_rules
from uc3.sheet import (
    load_sheet,
    load_tests,
    recommend_tests,
    select_tests,
    sheet_rows_for_tests,
    test_selector_categories,
    test_selector_options,
)


st.set_page_config(
    page_title="MVB Compliance",
    page_icon=str(Path(__file__).parent / "assets" / "mvb-logo.png"),
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --mvb-navy: #12376a;
        --mvb-blue: #2f5f9f;
        --mvb-pale-blue: #eaf1f9;
        --mvb-surface: #ffffff;
        --mvb-grey: #e3e9f0;
        --mvb-muted: #66758a;
        --mvb-ink: #1d2b3d;
    }

    .stApp {
        background: #f3f6fa;
        color: var(--mvb-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(243, 246, 250, 0.92);
    }

    [data-testid="stStatusWidgetRunningIcon"] {
        position: relative;
        display: inline-block;
        width: 1.5rem;
        height: 1.5rem;
    }

    [data-testid="stStatusWidgetRunningIcon"] svg,
    [data-testid="stStatusWidgetRunningIcon"] img {
        visibility: hidden;
    }

    [data-testid="stStatusWidgetRunningIcon"]::after {
        position: absolute;
        inset: 3px;
        border: 2px solid #d8e1eb;
        border-top-color: var(--mvb-blue);
        border-radius: 50%;
        content: "";
        animation: mvb-status-spin 0.8s linear infinite;
    }

    @keyframes mvb-status-spin {
        to {
            transform: rotate(360deg);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        [data-testid="stStatusWidgetRunningIcon"]::after {
            animation: none;
            border-color: var(--mvb-blue);
        }
    }

    [data-testid="stSidebar"] {
        background: #e8edf4;
        border-right: 1px solid #d5dee9;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #d8e1eb;
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
    }

    .mvb-brand-name {
        color: var(--mvb-navy);
        font-size: 1.02rem;
        font-weight: 750;
        line-height: 1.2;
    }

    .mvb-brand-subtitle {
        margin-top: 0.2rem;
        color: var(--mvb-muted);
        font-size: 0.75rem;
    }

    h1, h2, h3 {
        color: var(--mvb-navy);
        letter-spacing: -0.015em;
    }

    .stApp h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.95rem;
        font-weight: 600;
    }

    .stApp h2 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.45rem;
        font-weight: 600;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.45rem;
        border-bottom: 1px solid #d8e1eb;
    }

    [data-testid="stTabs"] button[role="tab"] {
        min-height: 2.8rem;
        border-radius: 10px 10px 0 0;
        color: #52647a;
        font-weight: 650;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #e8eef7;
        color: var(--mvb-navy);
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
        color: var(--mvb-navy);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid #dfe5ec;
        border-radius: 10px;
        background: #fff;
        box-shadow: none;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-color: #d3deed;
        background: #edf2f8;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stMultiSelect"] > div > div,
    [data-testid="stFileUploader"] section {
        border-color: #ccd6e2;
        border-radius: 11px;
    }

    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.72);
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--mvb-blue);
    }

    [data-testid="stBaseButton-primary"],
    [data-testid="stDownloadButton"] button {
        border: 1px solid var(--mvb-navy);
        border-radius: 10px;
        background: var(--mvb-navy);
        color: #fff;
        font-weight: 650;
        transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }

    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color: var(--mvb-blue);
        background: var(--mvb-blue);
        color: #fff;
        transform: translateY(-1px);
    }

    [data-testid="stBaseButton-secondary"] {
        border-color: #cbd6e3;
        border-radius: 8px;
        color: var(--mvb-navy);
    }

    [data-testid="stMetric"] {
        border: 1px solid #dce4ed;
        border-radius: 9px;
        background: var(--mvb-surface);
        box-shadow: none;
        padding: 0.85rem 1rem;
    }

    [data-testid="stMetricLabel"] p {
        color: var(--mvb-muted);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--mvb-navy);
    }

    [data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid #dce4ed;
        border-radius: 8px;
        background: #fff;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid #dce4ed;
        border-radius: 8px;
        background: #fff;
    }

    .prompt-label {
        margin: 1.2rem 0 0.4rem;
        color: #52647a;
        font-size: 0.82rem;
        font-weight: 650;
    }

    div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"] > hr) {
        border-color: #d8e1eb;
    }

    :focus-visible {
        outline: 3px solid rgba(47, 95, 159, 0.45) !important;
        outline-offset: 2px;
    }

    @media (max-width: 700px) {
        [data-testid="stChatMessage"] {
            padding: 0.75rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources(config_path):
    with Path(config_path).open(encoding="utf8") as file:
        config = yaml.safe_load(file)

    if config.get("chunks_file"):
        documents = load_chunked_documents(config["chunks_file"])
    else:
        documents = load_documents(
            config["documents_dir"],
            config["retrieval"]["chunk_max_chars"],
        )

    return config, documents, make_backend(config["llm"])


def _file_signature(files):
    digest = hashlib.sha256()
    for uploaded in files:
        digest.update(uploaded.name.encode("utf8"))
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def _safe_filename(name):
    return Path(name.replace("\\", "/")).name


def _corpus_signature(documents):
    digest = hashlib.sha256()
    for code, document in sorted(documents.items()):
        digest.update(code.encode("utf8"))
        for chunk in document.chunks:
            digest.update(
                f"{chunk.doc_code}\0{chunk.page_start}\0{chunk.page_end}\0"
                f"{chunk.heading}\0{chunk.text}\0".encode("utf8")
            )
    return digest.hexdigest()


config, base_documents, backend = load_resources("config.yaml")
model_name = getattr(backend, "model", config["llm"]["model"])

if "upload_root" not in st.session_state:
    st.session_state.upload_root = tempfile.mkdtemp(prefix="uc3-streamlit-")
upload_root = Path(st.session_state.upload_root)

conversation_tab, results_tab, documents_tab = st.tabs(
    ["Conversation", "Testing", "Documents"]
)

# The batch tab is processed first so the uploaded PDF corpus is also available
# to the conversation tab on the same Streamlit run.
with results_tab:
    st.title("Test verification results")
    st.caption(
        "Upload PDFs, select test IDs from the configured workbook, "
        "then download a detailed Excel report."
    )

    uploaded_pdfs = st.file_uploader(
        "Upload PDF documents for this run",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_pdf_uploads",
        help=(
            "PDFs are combined with the indexed corpus. A matching document "
            "code in the filename replaces that corpus document."
        ),
    )
    pdf_signature = _file_signature(uploaded_pdfs)
    if pdf_signature != st.session_state.get("pdf_signature"):
        uploaded_documents = {}
        upload_errors = []
        if uploaded_pdfs:
            corpus_dir = upload_root / f"corpus-{pdf_signature}"
            corpus_dir.mkdir(parents=True, exist_ok=True)
            for index, uploaded in enumerate(uploaded_pdfs):
                filename = _safe_filename(uploaded.name)
                target = corpus_dir / filename
                if target.exists():
                    target = corpus_dir / f"{index}_{filename}"
                target.write_bytes(uploaded.getvalue())
                try:
                    document = load_document(
                        target,
                        config["retrieval"]["chunk_max_chars"],
                    )
                except (OSError, ValueError, PDFSyntaxError) as error:
                    upload_errors.append(f"{uploaded.name}: {error}")
                    continue
                if document.code in uploaded_documents:
                    upload_errors.append(
                        f"Multiple uploads map to {document.code}; "
                        f"using {uploaded.name}."
                    )
                uploaded_documents[document.code] = document
        st.session_state.pdf_signature = pdf_signature
        st.session_state.uploaded_documents = uploaded_documents
        st.session_state.upload_errors = upload_errors

    uploaded_documents = st.session_state.get("uploaded_documents", {})
    documents = {**base_documents, **uploaded_documents}
    for upload_error in st.session_state.get("upload_errors", []):
        st.warning(upload_error)

    replaced_codes = sorted(set(base_documents) & set(uploaded_documents))
    st.caption(
        f"{len(documents)} documents available "
        f"({len(uploaded_documents)} uploaded for this session)."
    )
    if replaced_codes:
        st.info("Uploads replace documents with matching codes: " + ", ".join(replaced_codes))

    corpus_signature = _corpus_signature(documents)
    if st.session_state.get("corpus_signature") != corpus_signature:
        with st.spinner("Indexing document passages for search..."):
            st.session_state.corpus_index = CorpusIndex(documents.values())
            st.session_state.corpus_signature = corpus_signature

    workbook_path = Path(config["tests_file"])
    try:
        sheet_df = load_sheet(workbook_path)
        all_tests = load_tests(workbook_path)
    except (OSError, ValueError, ImportError, KeyError) as error:
        st.error(f"Could not read the configured test workbook: {error}")
        all_tests = []
        sheet_df = None
    else:
        test_ids, test_labels = test_selector_options(sheet_df, all_tests)
        test_categories = test_selector_categories(sheet_df, all_tests)
        st.caption(f"Test source: `{workbook_path}` · {len(test_ids)} available test IDs")
        recommendations = recommend_tests(
            all_tests,
            uploaded_documents.keys(),
            documents.keys(),
        )
        recommended_ids = [test.test_id for test in recommendations]
        if uploaded_documents:
            if recommended_ids:
                st.info(
                    f"Recommended {len(recommended_ids)} test(s) based on the "
                    "document codes in your uploaded PDFs. Recommendations are "
                    "matched from the workbook, not generated by the model."
                )
                st.caption(
                    "Suggested IDs: " + ", ".join(recommended_ids)
                )
            else:
                st.warning(
                    "No tests matched the uploaded document codes with all "
                    "required documents available. You can still select tests manually."
                )
        else:
            st.caption("Upload PDFs to get suggested test IDs.")
        selection_state_key = f"selected_tests_{pdf_signature}"
        if selection_state_key not in st.session_state:
            st.session_state[selection_state_key] = recommended_ids
        selected_category = st.selectbox(
            "Choose a test category",
            options=list(test_categories),
            key=f"test_category_{pdf_signature}",
        )
        category_test_ids = test_categories[selected_category]
        category_selection_key = (
            f"selected_test_ids_{pdf_signature}_{selected_category}"
        )
        st.session_state[category_selection_key] = [
            test_id for test_id in st.session_state[selection_state_key]
            if test_id in category_test_ids
        ]

        def save_category_selection():
            selected = set(st.session_state[selection_state_key])
            selected.difference_update(category_test_ids)
            selected.update(st.session_state[category_selection_key])
            st.session_state[selection_state_key] = [
                test_id for test_id in test_ids if test_id in selected
            ]

        selected_ids = st.multiselect(
            f"Select tests in {selected_category}",
            options=category_test_ids,
            format_func=lambda test_id: test_labels.get(test_id, test_id),
            placeholder="Choose one or more tests",
            key=category_selection_key,
            on_change=save_category_selection,
        )
        selected_ids = st.session_state[selection_state_key]
        st.caption(f"{len(selected_ids)} test(s) selected across categories.")
        selected = select_tests(all_tests, selected_ids)

        input_signature = hashlib.sha256(
            f"{corpus_signature}\0{workbook_path.resolve()}\0"
            f"{','.join(test.test_id for test in selected)}".encode("utf8")
        ).hexdigest()

        if st.button(
            "Run selected tests",
            type="primary",
            disabled=not selected,
        ):
            try:
                rules = load_rules(config["rules_file"])
                as_of = date.fromisoformat(
                    config.get("as_of_date") or date.today().isoformat()
                )
                with st.spinner(f"Evaluating {len(selected)} selected tests..."):
                    results, _ = evaluate_tests(
                        selected,
                        documents,
                        backend,
                        rules,
                        config,
                        as_of,
                    )
                    output_path = upload_root / "latest-results.xlsx"
                    write_results(
                        sheet_rows_for_tests(sheet_df, selected),
                        selected,
                        results,
                        output_path,
                        model_name,
                        as_of,
                        config,
                    )
                st.session_state.batch_results = results
                st.session_state.batch_excel = output_path.read_bytes()
                st.session_state.batch_signature = input_signature
            except (OSError, ValueError, requests.RequestException) as error:
                st.error(f"Could not complete the verification run: {error}")

        if st.session_state.get("batch_signature") == input_signature:
            results = st.session_state.get("batch_results", [])
            if results:
                counts = {
                    verdict: sum(result.verdict == verdict for result in results)
                    for verdict in ("PASS", "FAIL", "UNCLEAR")
                }
                col_pass, col_fail, col_unclear = st.columns(3)
                col_pass.metric("PASS", counts["PASS"])
                col_fail.metric("FAIL", counts["FAIL"])
                col_unclear.metric("UNCLEAR", counts["UNCLEAR"])

                st.dataframe(
                    [{
                        "Test": result.test_id,
                        "Result": result.verdict,
                        "Reason": result.reason,
                        "Documents": ", ".join(result.documents),
                        "Review required": "Yes" if result.review_flag else "No",
                    } for result in results],
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download Excel results",
                    data=st.session_state.batch_excel,
                    file_name="compliance-results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        elif st.session_state.get("batch_results"):
            st.info("Inputs or test selection changed. Run the selected tests again.")

with st.sidebar:
    brand_image, brand_name = st.columns([1, 3], vertical_alignment="center")
    with brand_image:
        st.image(
            Path(__file__).parent / "assets" / "mvb-logo.png",
            width=56,
            caption=None,
        )
    with brand_name:
        st.markdown(
            """
            <div class="mvb-brand-name">MVB Compliance</div>
            <div class="mvb-brand-subtitle">A workspace for careful review</div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption(f"Model configured: {model_name}")
    st.caption("Review answers and test verdicts against source documents.")

with conversation_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Conversation")
    st.caption("Ask questions about indexed documents.")

    if not st.session_state.messages:
        st.markdown('<div class="prompt-label">A few places to start</div>', unsafe_allow_html=True)
        prompt_columns = st.columns(3)
        example_prompts = [
            "For [client name], how often are access reviews performed?",
            "For [client name], what does the sanctions policy require?",
            "For [client name], summarize the fraud escalation process.",
        ]
        for column, prompt in zip(prompt_columns, example_prompts):
            with column:
                if st.button(prompt, key=f"prompt_{prompt}", use_container_width=True):
                    st.session_state.pending_question = prompt
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                with st.expander(
                    f"{citation['document']} · page {citation['page']}"
                ):
                    st.write(citation["quote"])

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask a question about the documents...")
    question = pending_question or question
    if question:
        history = [
            {"role": item["role"], "content": item["content"]}
            for item in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating an answer..."):
                    started = time.perf_counter()
                    response = answer_question(
                        question,
                        documents,
                        backend,
                        top_k=config["retrieval"]["top_k"],
                        history=history,
                        corpus_index=st.session_state.corpus_index,
                    )
            except (requests.RequestException, ValueError) as error:
                st.error(f"Could not get an answer: {error}")
            else:
                st.markdown(response.answer)
                st.caption(f"Completed in {time.perf_counter() - started:.1f} seconds")
                for citation in response.citations:
                    with st.expander(
                        f"{citation['document']} · page {citation['page']}"
                    ):
                        st.write(citation["quote"])
                for note in response.notes:
                    st.caption(note)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "citations": response.citations,
                })

with documents_tab:
    st.title("Indexed documents")
    st.caption(
        "Browse the text passages currently available to chat and test runs."
    )

    if not documents:
        st.info("No documents are currently indexed.")
    else:
        search = st.text_input(
            "Find a document",
            placeholder="Search by document code, title, or filename",
            key="document_search",
        ).strip().casefold()
        matching_codes = [
            code for code, document in sorted(documents.items())
            if search in " ".join((
                code,
                str(document.header.get("title", "")),
                Path(document.path).name,
            )).casefold()
        ]

        if not matching_codes:
            st.info("No indexed documents match that search.")
        else:
            selected_code = st.selectbox(
                "Choose a document",
                matching_codes,
                format_func=lambda code: (
                    f"{code} - "
                    f"{documents[code].header.get('title') or Path(documents[code].path).name}"
                ),
                key="document_preview_code",
            )
            document = documents[selected_code]
            source_name = Path(document.path).name

            col_code, col_pages, col_chunks = st.columns(3)
            col_code.metric("Document code", document.code)
            col_pages.metric("Pages", len(document.pages))
            col_chunks.metric("Indexed passages", len(document.chunks))
            st.caption(f"Source file: {source_name} · {document.chars:,} extracted characters")

            if document.header:
                st.subheader("Document metadata")
                st.json(document.header, expanded=True)
            if document.doc_date:
                st.caption(
                    f"Detected document date: {document.doc_date.isoformat()}"
                    + (f" · {document.date_source}" if document.date_source else "")
                )

            st.subheader("Indexed passages")
            for index, chunk in enumerate(document.chunks, start=1):
                page_label = (
                    str(chunk.page_start)
                    if chunk.page_start == chunk.page_end
                    else f"{chunk.page_start}-{chunk.page_end}"
                )
                heading = chunk.heading or "Unlabeled section"
                with st.expander(
                    f"{index}. {heading} · page {page_label}",
                    expanded=(index == 1),
                ):
                    st.text(chunk.text)

            if config.get("chunks_file") and selected_code in base_documents:
                st.caption(
                    "This corpus document was loaded from the supplied chunk file. "
                    "Its page references use that file's section-start page metadata; "
                    "verify exact locations in the original PDF."
                )
