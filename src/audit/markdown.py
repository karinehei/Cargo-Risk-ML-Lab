"""Markdown tables with an explicit header cell for every column."""

from __future__ import annotations


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a GitHub-flavoured Markdown table.

    Raises:
        ValueError: If a header is missing/blank or a row has the wrong width.
    """
    if not headers:
        raise ValueError("A Markdown table needs at least one column header")
    cleaned = [header.strip() for header in headers]
    if any(not header for header in cleaned):
        raise ValueError("Every column must have a separate non-empty header")
    for row in rows:
        if len(row) != len(cleaned):
            raise ValueError(
                f"Row has {len(row)} cells but there are {len(cleaned)} column headers"
            )
    header_line = "| " + " | ".join(cleaned) + " |"
    separator = "| " + " | ".join("---" for _ in cleaned) + " |"
    body = ["| " + " | ".join(cell for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])
