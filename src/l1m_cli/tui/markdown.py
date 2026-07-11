from __future__ import annotations

import re


def render_markdown_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_code = False
    code_lang = ""
    source_lines = text.splitlines() or [text]
    index = 0

    while index < len(source_lines):
        raw_line = source_lines[index]
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if in_code:
                lines.append(f"end code {code_lang}".rstrip())
                in_code = False
                code_lang = ""
            else:
                code_lang = stripped[3:].strip()
                lines.append(f"code {code_lang}".rstrip())
                in_code = True
            index += 1
            continue

        if in_code:
            lines.append(f"  {raw_line.rstrip()}")
            index += 1
            continue

        if _is_table_start(source_lines, index):
            rendered_table, index = _render_table(source_lines, index)
            lines.extend(rendered_table)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", raw_line)
        if heading:
            title = _render_inline(heading.group(2)).strip()
            lines.append(title)
            lines.append("-" * min(max(len(title), 8), 48))
            index += 1
            continue

        quote = re.match(r"^\s*>\s?(.*)$", raw_line)
        if quote:
            lines.append(f"| {_render_inline(quote.group(1))}")
            index += 1
            continue

        bullet = re.match(r"^(\s*)[-*+]\s+(.+)$", raw_line)
        if bullet:
            indent = " " * (len(bullet.group(1)) // 2 * 2)
            lines.append(f"{indent}- {_render_inline(bullet.group(2))}")
            index += 1
            continue

        ordered = re.match(r"^(\s*)(\d+)[.)]\s+(.+)$", raw_line)
        if ordered:
            indent = " " * (len(ordered.group(1)) // 2 * 2)
            lines.append(f"{indent}{ordered.group(2)}. {_render_inline(ordered.group(3))}")
            index += 1
            continue

        lines.append(_render_inline(raw_line))
        index += 1

    return lines


def _render_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and _is_table_row(lines[index])
        and _is_table_separator(lines[index + 1])
    )


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    cells = _parse_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _parse_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_render_inline(cell.strip()) for cell in stripped.split("|")]


def _render_table(lines: list[str], start: int) -> tuple[list[str], int]:
    headers = _parse_table_row(lines[start])
    index = start + 2
    rows: list[list[str]] = []

    while index < len(lines) and _is_table_row(lines[index]):
        row = _parse_table_row(lines[index])
        if row:
            rows.append(row)
        index += 1

    return _table_to_list(headers, rows), index


def _table_to_list(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not headers or not rows:
        return []

    width = max([len(headers), *(len(row) for row in rows)])
    normalized_headers = [
        (headers[column].strip() if column < len(headers) and headers[column].strip() else f"col{column + 1}")
        for column in range(width)
    ]
    result_index = _find_result_column(normalized_headers)
    rendered: list[str] = []

    for row in rows:
        cells = [row[column].strip() if column < len(row) else "" for column in range(width)]
        if result_index is not None:
            rendered.append(_render_result_row(normalized_headers, cells, result_index))
        else:
            rendered.append(_render_generic_row(normalized_headers, cells))

    return [line for line in rendered if line != "- "]


def _find_result_column(headers: list[str]) -> int | None:
    result_headers = {"result", "status", "outcome", "\u7ed3\u679c", "\u72b6\u6001"}
    for index, header in enumerate(headers):
        if header.strip().lower() in result_headers:
            return index
    return None


def _render_result_row(headers: list[str], cells: list[str], result_index: int) -> str:
    prefix = ""
    main_parts: list[str] = []

    for index, cell in enumerate(cells):
        if index == result_index or not cell:
            continue
        if _is_index_header(headers[index]):
            prefix = cell
        else:
            main_parts.append(cell)

    main = " / ".join(main_parts) or " / ".join(
        cell for index, cell in enumerate(cells) if index != result_index and cell
    )
    if prefix:
        main = f"{prefix} {main}".strip()

    result = cells[result_index]
    return f"- {main}: {result}" if result else f"- {main}"


def _render_generic_row(headers: list[str], cells: list[str]) -> str:
    pairs = [
        f"{headers[index]}: {cell}"
        for index, cell in enumerate(cells)
        if cell
    ]
    return "- " + "; ".join(pairs)


def _is_index_header(header: str) -> bool:
    return header.strip().lower() in {"#", "no", "no.", "index", "\u5e8f\u53f7", "\u7f16\u53f7"}
