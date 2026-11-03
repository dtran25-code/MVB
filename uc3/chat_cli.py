"""Try the document chat backend in a terminal: python -m uc3.chat_cli."""
import argparse
from pathlib import Path
import yaml

from .chat import answer_question
from .documents import load_chunked_documents, load_documents
from .llm import make_backend
from .retrieve import CorpusIndex


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask questions about the indexed documents")
    parser.add_argument("--config", default="config.yaml", help="path to the YAML configuration")
    parser.add_argument("--model", help="override the configured model name")
    parser.add_argument("--question", help="ask one question, then exit")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    with config_path.open(encoding="utf8") as file:
        cfg = yaml.safe_load(file)
    if args.model:
        cfg["llm"]["model"] = args.model

    if cfg.get("chunks_file"):
        documents = load_chunked_documents(cfg["chunks_file"])
    else:
        documents = load_documents(
            cfg["documents_dir"],
            cfg["retrieval"]["chunk_max_chars"],
        )
    backend = make_backend(cfg["llm"])
    top_k = cfg["retrieval"]["top_k"]
    corpus_index = CorpusIndex(documents.values())

    print(f"Loaded {len(documents)} document(s); model: {getattr(backend, 'model', cfg['llm']['model'])}")
    print("Citations are checked against retrieved text; review answers against the source documents.")

    history = []
    if args.question:
        questions = [args.question]
    else:
        print("Type a question, or 'quit' to exit.")
        questions = None

    while True:
        question = questions[0] if questions is not None else input("\nYou: ").strip()
        if questions is not None:
            questions.pop(0)
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            if questions is not None:
                break
            continue

        response = answer_question(
            question,
            documents,
            backend,
            top_k,
            history,
            corpus_index=corpus_index,
        )
        print(f"\nAssistant: {response.answer}")
        for citation in response.citations:
            print(
                f"  [{citation['document']} p.{citation['page']}] "
                f"\"{citation['quote']}\""
            )
        for note in response.notes:
            print(f"  Note: {note}")

        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response.answer},
        ])
        if questions is not None:
            break


if __name__ == "__main__":
    main()
