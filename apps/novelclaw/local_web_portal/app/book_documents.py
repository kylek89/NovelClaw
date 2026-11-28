"""Author-maintained Book documents. Each edit is an immutable revision."""
from difflib import ndiff
import re
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .models import BookDocumentRevision as Revision

TITLES = {"premise": "Story Premise", "author_brief": "Author Brief"}


def explicit_addition(reply):
    match = re.fullmatch(
        r"(?:please\s+)?(?:add|save|remember)(?:\s+this)?\s+(?:to|in|for)\s+(?:the\s+|my\s+)?"
        r"(story premise|author brief)\s*:\s*(.+)", reply.strip(), re.I | re.S,
    )
    if match:
        return ("premise" if match[1].casefold() == "story premise" else "author_brief", match[2].strip())
    return None


def revisions(book_id, kind):
    return select(Revision).where(
        Revision.book_id == book_id, Revision.kind == kind
    ).order_by(Revision.version.desc())


def current(db, book_id, kind):
    return db.scalar(revisions(book_id, kind).limit(1))


def changes(history):
    return {row.version: "\n".join(line for line in ndiff(
        history[i + 1].content.splitlines() if i + 1 < len(history) else [],
        row.content.splitlines(),
    ) if line.startswith(("+ ", "- "))) for i, row in enumerate(history)}


def versions(db, book_id):
    return {kind: row.version if (row := current(db, book_id, kind)) else 0 for kind in TITLES}


def save(db, book_id, kind, expected, text, chat=None):
    if kind not in TITLES:
        raise HTTPException(404, "Unknown Book document.")
    row = current(db, book_id, kind)
    text = text.replace("\r\n", "\n").strip()
    if len(text) > 24000:
        raise HTTPException(422, "Keep each Book document under 24,001 characters.")
    conflict = f"{TITLES[kind]} changed. Open its current version before saving."
    if expected != (row.version if row else 0):
        raise HTTPException(409, conflict)
    if text == (row.content if row else ""):
        return row
    result = Revision(book_id=book_id, kind=kind, version=expected + 1, content=text, source_chat=chat)
    db.add(result)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, conflict) from exc
    return result


def packet(db, book_id):
    sections = []
    for kind, title in TITLES.items():
        row = current(db, book_id, kind)
        if row:
            sections.append(f"[{title} · revision {row.version}]\n{row.content or '(empty)'}")
    return ("[Current author-approved Book documents; take precedence over chat summaries]\n"
            + "\n\n".join(sections)) if sections else ""


def add_answers(db, book_id, chat, additions, expected):
    saved = {}
    for kind, answers in additions.items():
        if kind not in TITLES or not isinstance(expected, dict) or type(expected.get(kind)) is not int:
            raise HTTPException(422, "Refresh the chat to load the current Book document.")
        row = current(db, book_id, kind)
        paragraphs = row.content.split("\n\n") if row and row.content else []
        for answer in answers:
            if answer in paragraphs:
                continue
            question, separator, _ = answer.partition(" — ")
            if separator and any(line.startswith(question + separator) for line in paragraphs):
                raise HTTPException(409, f"That question already has a different answer in {TITLES[kind]}. "
                                    "Edit the document to replace it, or keep this answer in chat.")
            paragraphs.append(answer)
        result = save(db, book_id, kind, expected[kind], "\n\n".join(paragraphs), chat)
        if result and result.version != expected[kind]:
            saved[kind] = result.version
    return saved
