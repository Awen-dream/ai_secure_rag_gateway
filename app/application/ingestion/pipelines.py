import re


def chunk_text(content: str, max_words: int = 140) -> list[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", content) if segment.strip()]
    if not paragraphs:
        return [content.strip()] if content.strip() else []

    chunks: list[str] = []
    buffer: list[str] = []
    for paragraph in paragraphs:
        if buffer and len(" ".join(buffer + [paragraph]).split()) > max_words:
            chunks.append("\n\n".join(buffer))
            buffer = [paragraph]
            continue
        buffer.append(paragraph)

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks
