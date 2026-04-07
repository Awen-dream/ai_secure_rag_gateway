from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.ingestion.pipelines import chunk_document
from app.application.ingestion.tokenization import estimate_token_count
from app.core.config import settings


def _preview(text: str, limit: int = 80) -> str:
    single_line = " ".join(text.strip().split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 3]}..."


def _read_inputs(paths: list[str], stdin_text: str) -> list[tuple[str, str]]:
    inputs: list[tuple[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        inputs.append((str(path), path.read_text(encoding="utf-8")))

    if stdin_text.strip():
        inputs.append(("<stdin>", stdin_text))

    return inputs


def _iter_chunk_lines(text: str, max_tokens: int, overlap_tokens: int) -> Iterable[str]:
    chunks = chunk_document(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    yield f"chunks={len(chunks)}"
    for index, chunk in enumerate(chunks, start=1):
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else "Document"
        yield (
            f"  [{index}] tokens={chunk.token_count} "
            f"section={heading!r} preview={_preview(chunk.text)!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect token counts and chunk sizes using the project's current tokenizer and chunking rules."
    )
    parser.add_argument("paths", nargs="*", help="UTF-8 text files to inspect.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.chunk_max_tokens,
        help=f"Chunk size budget. Default: {settings.chunk_max_tokens}.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=settings.chunk_overlap_tokens,
        help=f"Chunk overlap budget. Default: {settings.chunk_overlap_tokens}.",
    )
    args = parser.parse_args()

    stdin_text = ""
    if not args.paths:
        if sys.stdin.isatty():
            print("Paste text, then press Ctrl-D when done:")
        stdin_text = sys.stdin.read()

    inputs = _read_inputs(args.paths, stdin_text)
    if not inputs:
        raise SystemExit("No input provided. Pass file paths or pipe/paste text.")

    print(f"tokenizer_model={settings.chunk_tokenizer_model}")
    print(f"tokenizer_encoding={settings.chunk_tokenizer_encoding or '<model-default>'}")
    print(f"chunk_max_tokens={args.max_tokens}")
    print(f"chunk_overlap_tokens={args.overlap_tokens}")

    for label, text in inputs:
        total_tokens = estimate_token_count(text)
        print()
        print(f"== {label} ==")
        print(f"characters={len(text)}")
        print(f"tokens={total_tokens}")
        print(f"preview={_preview(text, limit=120)!r}")
        for line in _iter_chunk_lines(text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens):
            print(line)


if __name__ == "__main__":
    main()
