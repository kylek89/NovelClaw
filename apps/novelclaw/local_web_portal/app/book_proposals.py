"""Validate structured writing output before exposing selected candidates or changing outlines."""

import json
import re

from pydantic import BaseModel, Field

from .book_data import Chapter, Text, remember


class OutlineDraft(BaseModel):
    chapter: int = Field(ge=1, le=999)
    outline: Text = Field(min_length=1)


def parse_rows(output, kind):
    value = output.strip()
    if value.startswith("\x60\x60\x60"):
        value = re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", value)
    rows = json.loads(value)
    if not isinstance(rows, list) or len(rows) > 80:
        raise ValueError("Invalid proposal list")
    return [kind.model_validate(row) for row in rows]


def save_outlines(book, output, first, last):
    rows = parse_rows(output, OutlineDraft)
    if [row.chapter for row in rows] != list(range(first, last + 1)):
        raise ValueError("Expected one outline for each chapter in order")
    if first > len(book.chapters) + 1:
        raise ValueError("Choose a contiguous chapter range")
    for row in rows:
        if row.chapter > len(book.chapters):
            book.chapters.append(Chapter(title=f"Chapter {row.chapter}"))
        ch = book.chapters[row.chapter - 1]
        remember(book, "outline", ch.id, ch.outline, row.outline)
        ch.outline = row.outline
