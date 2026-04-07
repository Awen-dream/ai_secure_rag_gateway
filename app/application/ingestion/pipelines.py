from __future__ import annotations

import re
from dataclasses import dataclass

from app.application.ingestion.tokenization import estimate_token_count
from app.core.config import settings

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])|(?<=[.?!;])\s+")
_NATURAL_BREAK_RE = re.compile(r"[，,、。！？!?；;\s]")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_HEADING_RE = re.compile(r"^(=+|-+)\s*$")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<prefix>(?:\d+(?:\.\d+)*[.)]?|[一二三四五六七八九十百千万]+[、.．]|第[一二三四五六七八九十百千万0-9]+[章节部分条]))\s*(?P<title>.+)$"
)
_BULLET_LIST_RE = re.compile(r"^[-*+]\s+.+$")
_ORDERED_LIST_RE = re.compile(r"^\d+[.)]\s+.+$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")


@dataclass(frozen=True)
class ChunkPayload:
    text: str
    section_name: str
    heading_path: list[str]
    token_count: int


@dataclass(frozen=True)
class _Block:
    kind: str
    text: str


@dataclass(frozen=True)
class _Section:
    heading_path: list[str]
    blocks: list[_Block]


def _resolve_chunk_limits(
    max_tokens: int | None,
    overlap_tokens: int | None,
) -> tuple[int, int]:
    resolved_max_tokens = max_tokens if max_tokens is not None else settings.chunk_max_tokens
    resolved_overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens

    resolved_max_tokens = max(int(resolved_max_tokens), 1)
    resolved_overlap_tokens = max(int(resolved_overlap_tokens), 0)
    if resolved_max_tokens == 1:
        return resolved_max_tokens, 0

    return resolved_max_tokens, min(resolved_overlap_tokens, resolved_max_tokens - 1)


def _normalize_heading_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().strip("#").strip()


def _infer_numbered_heading_level(prefix: str) -> int:
    if prefix.startswith("第"):
        if prefix.endswith("章"):
            return 1
        if prefix.endswith("节"):
            return 2
        if prefix.endswith("条"):
            return 3
        return 1
    if any(marker in prefix for marker in ("、", "．", ".")):
        dots = prefix.rstrip(".)")
        if dots and dots[0].isdigit():
            return dots.count(".") + 1
        return 1
    return 1


def _match_heading_line(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None

    markdown_match = _MARKDOWN_HEADING_RE.match(stripped)
    if markdown_match:
        return len(markdown_match.group(1)), _normalize_heading_title(markdown_match.group(2))

    if len(stripped) > 120 or stripped.endswith(("。", "！", "？", ";", "；")):
        return None

    numbered_match = _NUMBERED_HEADING_RE.match(stripped)
    if numbered_match:
        title = _normalize_heading_title(numbered_match.group("title"))
        if title:
            return _infer_numbered_heading_level(numbered_match.group("prefix")), title

    return None


def _match_setext_heading(current_line: str, next_line: str | None) -> tuple[int, str] | None:
    if next_line is None:
        return None

    underline = next_line.strip()
    if not _SETEXT_HEADING_RE.match(underline):
        return None

    title = _normalize_heading_title(current_line)
    if not title:
        return None
    level = 1 if underline.startswith("=") else 2
    return level, title


def _is_code_fence(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _is_list_item(line: str) -> bool:
    stripped = line.strip()
    return bool(_BULLET_LIST_RE.match(stripped) or _ORDERED_LIST_RE.match(stripped))


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(_TABLE_SEPARATOR_RE.match(lines[index + 1].strip()))


def _update_heading_stack(
    stack: list[tuple[int, str]],
    level: int,
    title: str,
) -> list[tuple[int, str]]:
    updated = list(stack)
    while updated and updated[-1][0] >= level:
        updated.pop()
    updated.append((level, title))
    return updated


def _append_paragraph(blocks: list[_Block], paragraph_lines: list[str]) -> None:
    paragraph = "\n".join(line.rstrip() for line in paragraph_lines).strip()
    if paragraph:
        blocks.append(_Block(kind="paragraph", text=paragraph))
    paragraph_lines.clear()


def _append_block(blocks: list[_Block], kind: str, lines: list[str]) -> None:
    text = "\n".join(line.rstrip() for line in lines).strip()
    if text:
        blocks.append(_Block(kind=kind, text=text))


def _append_section(
    sections: list[_Section],
    heading_stack: list[tuple[int, str]],
    blocks: list[_Block],
) -> None:
    if not blocks:
        return
    sections.append(_Section(heading_path=[title for _, title in heading_stack], blocks=list(blocks)))
    blocks.clear()


def _collect_code_block(lines: list[str], start: int) -> tuple[list[str], int]:
    block_lines = [lines[start]]
    index = start + 1
    while index < len(lines):
        block_lines.append(lines[index])
        if _is_code_fence(lines[index]):
            return block_lines, index + 1
        index += 1
    return block_lines, index


def _collect_table_block(lines: list[str], start: int) -> tuple[list[str], int]:
    block_lines = [lines[start], lines[start + 1]]
    index = start + 2
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or "|" not in lines[index]:
            break
        block_lines.append(lines[index])
        index += 1
    return block_lines, index


def _collect_list_block(lines: list[str], start: int) -> tuple[list[str], int]:
    block_lines = [lines[start]]
    index = start + 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            break
        if _is_code_fence(lines[index]) or _is_table_start(lines, index) or _match_heading_line(lines[index]):
            break
        if _is_list_item(lines[index]) or lines[index].startswith(("  ", "\t")):
            block_lines.append(lines[index])
            index += 1
            continue
        break
    return block_lines, index


def _split_sections(content: str) -> list[_Section]:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    blocks: list[_Block] = []
    paragraph_lines: list[str] = []

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        next_line = lines[index + 1] if index + 1 < len(lines) else None

        setext_heading = _match_setext_heading(line, next_line)
        if setext_heading:
            _append_paragraph(blocks, paragraph_lines)
            _append_section(sections, heading_stack, blocks)
            heading_stack = _update_heading_stack(heading_stack, setext_heading[0], setext_heading[1])
            index += 2
            continue

        heading = _match_heading_line(line)
        if heading:
            _append_paragraph(blocks, paragraph_lines)
            _append_section(sections, heading_stack, blocks)
            heading_stack = _update_heading_stack(heading_stack, heading[0], heading[1])
            index += 1
            continue

        if not stripped:
            _append_paragraph(blocks, paragraph_lines)
            index += 1
            continue

        if _is_code_fence(line):
            _append_paragraph(blocks, paragraph_lines)
            code_block, index = _collect_code_block(lines, index)
            _append_block(blocks, "code", code_block)
            continue

        if _is_table_start(lines, index):
            _append_paragraph(blocks, paragraph_lines)
            table_block, index = _collect_table_block(lines, index)
            _append_block(blocks, "table", table_block)
            continue

        if _is_list_item(line):
            _append_paragraph(blocks, paragraph_lines)
            list_block, index = _collect_list_block(lines, index)
            _append_block(blocks, "list", list_block)
            continue

        paragraph_lines.append(line.strip())
        index += 1

    _append_paragraph(blocks, paragraph_lines)
    _append_section(sections, heading_stack, blocks)
    return sections


def _split_sentences(paragraph: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", paragraph).strip()
    if not collapsed:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(collapsed) if part.strip()]


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    segments: list[str] = []
    remaining = text.strip()

    while remaining:
        if estimate_token_count(remaining) <= max_tokens:
            segments.append(remaining)
            break

        lo = 1
        hi = len(remaining)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if estimate_token_count(remaining[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1

        cut = max(lo, 1)
        natural_breaks = list(_NATURAL_BREAK_RE.finditer(remaining[:cut]))
        if natural_breaks:
            natural_cut = natural_breaks[-1].end()
            if natural_cut >= max(1, cut // 2):
                cut = natural_cut

        segment = remaining[:cut].strip()
        if not segment:
            segment = remaining[:1]
            cut = 1

        segments.append(segment)
        remaining = remaining[cut:].strip()

    return segments


def _paragraph_units(paragraph: str, max_tokens: int) -> list[str]:
    if estimate_token_count(paragraph) <= max_tokens:
        return [paragraph]

    units: list[str] = []
    buffer = ""
    for sentence in _split_sentences(paragraph):
        if estimate_token_count(sentence) > max_tokens:
            if buffer:
                units.append(buffer)
                buffer = ""
            units.extend(_split_long_text(sentence, max_tokens))
            continue

        candidate = sentence if not buffer else f"{buffer} {sentence}"
        if estimate_token_count(candidate) <= max_tokens:
            buffer = candidate
            continue

        if buffer:
            units.append(buffer)
        buffer = sentence

    if buffer:
        units.append(buffer)

    return units or _split_long_text(paragraph, max_tokens)


def _chunk_lines(lines: list[str], max_tokens: int) -> list[str]:
    units: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        line_tokens = estimate_token_count(line)
        if line_tokens > max_tokens:
            if buffer:
                units.append("\n".join(buffer).strip())
                buffer = []
                buffer_tokens = 0
            units.extend(_split_long_text(line, max_tokens))
            continue

        if buffer and buffer_tokens + line_tokens > max_tokens:
            units.append("\n".join(buffer).strip())
            buffer = [line]
            buffer_tokens = line_tokens
            continue

        buffer.append(line)
        buffer_tokens += line_tokens

    if buffer:
        units.append("\n".join(buffer).strip())

    return [unit for unit in units if unit]


def _code_units(block_text: str, max_tokens: int) -> list[str]:
    if estimate_token_count(block_text) <= max_tokens:
        return [block_text]

    lines = block_text.splitlines()
    if len(lines) < 2 or not _is_code_fence(lines[0]):
        return _chunk_lines(lines, max_tokens)

    opening_fence = lines[0].rstrip()
    closing_fence = lines[-1].rstrip() if _is_code_fence(lines[-1]) else ""
    body_lines = lines[1:-1] if closing_fence else lines[1:]

    wrapper_tokens = estimate_token_count(opening_fence)
    if closing_fence:
        wrapper_tokens += estimate_token_count(closing_fence)
    body_budget = max(max_tokens - wrapper_tokens, 1)

    body_chunks = _chunk_lines(body_lines, body_budget)
    rendered: list[str] = []
    for chunk in body_chunks:
        parts = [opening_fence, chunk]
        if closing_fence:
            parts.append(closing_fence)
        rendered.append("\n".join(parts).strip())
    return rendered or [block_text]


def _block_units(block: _Block, max_tokens: int) -> list[str]:
    if block.kind == "paragraph":
        return _paragraph_units(block.text, max_tokens)
    if estimate_token_count(block.text) <= max_tokens:
        return [block.text]
    if block.kind == "code":
        return _code_units(block.text, max_tokens)
    return _chunk_lines(block.text.splitlines(), max_tokens)


def _tail_overlap_units(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []

    collected: list[str] = []
    token_total = 0
    for unit in reversed(units):
        collected.insert(0, unit)
        token_total += estimate_token_count(unit)
        if token_total >= overlap_tokens:
            break
    return collected


def _select_heading_prefix(heading_path: list[str], max_tokens: int) -> str:
    if not heading_path:
        return ""

    full_path = "\n".join(heading_path)
    if estimate_token_count(full_path) <= max(20, max_tokens // 3):
        return full_path

    leaf = heading_path[-1]
    if estimate_token_count(leaf) <= max(12, max_tokens // 5):
        return leaf

    return ""


def _chunk_section(section: _Section, max_tokens: int, overlap_tokens: int) -> list[ChunkPayload]:
    if not section.blocks:
        return []

    section_name = section.heading_path[-1] if section.heading_path else "Document"
    heading_prefix = _select_heading_prefix(section.heading_path, max_tokens)
    body_budget = max_tokens - estimate_token_count(heading_prefix) if heading_prefix else max_tokens
    if heading_prefix and body_budget < max(20, max_tokens // 2):
        heading_prefix = section.heading_path[-1]
        body_budget = max_tokens - estimate_token_count(heading_prefix)
    if heading_prefix and body_budget < max(20, max_tokens // 2):
        heading_prefix = ""
        body_budget = max_tokens

    body_budget = max(body_budget, 1)

    units: list[str] = []
    for block in section.blocks:
        units.extend(_block_units(block, body_budget))

    def render(body: str) -> str:
        return f"{heading_prefix}\n\n{body}".strip() if heading_prefix else body

    chunks: list[ChunkPayload] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for unit in units:
        unit_tokens = estimate_token_count(unit)
        if buffer and buffer_tokens + unit_tokens > body_budget:
            text = render("\n\n".join(buffer))
            chunks.append(
                ChunkPayload(
                    text=text,
                    section_name=section_name,
                    heading_path=list(section.heading_path),
                    token_count=estimate_token_count(text),
                )
            )
            buffer = _tail_overlap_units(buffer, overlap_tokens)
            buffer_tokens = sum(estimate_token_count(item) for item in buffer)

            while buffer and buffer_tokens + unit_tokens > body_budget:
                removed = buffer.pop(0)
                buffer_tokens -= estimate_token_count(removed)

        buffer.append(unit)
        buffer_tokens += unit_tokens

    if buffer:
        text = render("\n\n".join(buffer))
        chunks.append(
            ChunkPayload(
                text=text,
                section_name=section_name,
                heading_path=list(section.heading_path),
                token_count=estimate_token_count(text),
            )
        )

    return chunks


def chunk_document(
    content: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[ChunkPayload]:
    max_tokens, overlap_tokens = _resolve_chunk_limits(max_tokens, overlap_tokens)
    sections = _split_sections(content)
    if not sections:
        stripped = content.strip()
        if not stripped:
            return []
        return [
            ChunkPayload(
                text=stripped,
                section_name="Document",
                heading_path=[],
                token_count=estimate_token_count(stripped),
            )
        ]

    chunks: list[ChunkPayload] = []
    for section in sections:
        chunks.extend(_chunk_section(section, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return chunks


def chunk_text(
    content: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    return [chunk.text for chunk in chunk_document(content, max_tokens=max_tokens, overlap_tokens=overlap_tokens)]
