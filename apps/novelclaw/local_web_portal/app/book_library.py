"""Book catalog queries and permanent removal of one author's book."""

from urllib.parse import urlencode
from pathlib import Path
import re
import shutil

from fastapi import HTTPException
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy import delete, func, or_, select

from config import Config
from .models import Book, GenerationJob, IdeaCopilotSession, RetiredBookId
from .security import SESSION_SECRET
from .settings import BASE_DIR, RUNS_DIR

PAGE_SIZE = 12
confirmation = URLSafeTimedSerializer(SESSION_SECRET, salt="permanent-book-removal-v1")


def switch_tool(tool):
    if re.fullmatch(r"/console/[a-z_]+(?:/[a-z_]+){0,2}", tool or ""):
        return tool
    return "/console/chat"


def library_location(tool, search="", page=1):
    return "/books?" + urlencode(
        {"tool": switch_tool(tool), "q": search[:160], "page": max(1, page)}
    )


def catalog(db, user_id, search, page):
    statement = select(Book).where(Book.user_id == user_id)
    if search:
        statement = statement.where(
            or_(
                Book.title.icontains(search, autoescape=True),
                Book.series.icontains(search, autoescape=True),
            )
        )
    count = db.scalar(select(func.count()).select_from(statement.subquery()))
    pages = max(1, (count + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, pages))
    rows = list(
        db.scalars(
            statement.order_by(func.lower(Book.title), Book.id)
            .offset((page - 1) * PAGE_SIZE)
            .limit(PAGE_SIZE)
        )
    )
    return dict(items=rows, total=count, pages=pages, page=page)


def receipt(book):
    return [book.user_id, book.id, book.memory_key]


def _target(root, component):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,155}", component):
        raise HTTPException(409, "The book has an invalid storage identifier.")
    root = root.resolve()
    path = root / component
    if (
        path.is_symlink()
        or path.resolve() != path
        or (path.exists() and not path.is_dir())
    ):
        raise HTTPException(
            409, "A book storage location needs repair before deletion."
        )
    return path


def remove_book(db, book, token):
    try:
        verified = confirmation.loads(token, max_age=1800)
    except BadSignature:
        verified = None
    if verified != receipt(book):
        raise HTTPException(409, "Reopen the confirmation page for this book.")

    jobs = list(
        db.scalars(select(GenerationJob).where(GenerationJob.book_id == book.id))
    )
    if any(job.status in {"running", "queued"} for job in jobs):
        raise HTTPException(
            409, "Finish or cancel this book’s active generation before deleting it."
        )
    job_ids = [job.id for job in jobs]
    run_ids = {job.run_id for job in jobs if job.run_id}
    keys = run_ids | {book.memory_key}
    foreign_job = db.scalar(
        select(GenerationJob.id)
        .where(
            GenerationJob.book_id.is_distinct_from(book.id),
            GenerationJob.run_id.in_(keys),
        )
        .limit(1)
    )
    foreign_memory = db.scalar(
        select(Book.id).where(Book.id != book.id, Book.memory_key.in_(keys)).limit(1)
    )
    foreign_link = db.scalar(
        select(IdeaCopilotSession.id)
        .where(
            IdeaCopilotSession.book_id.is_distinct_from(book.id),
            IdeaCopilotSession.final_job_id.in_(job_ids),
        )
        .limit(1)
    )
    conflicting_chat = db.scalar(
        select(IdeaCopilotSession.id)
        .where(
            IdeaCopilotSession.book_id == book.id,
            IdeaCopilotSession.user_id != book.user_id,
        )
        .limit(1)
    )
    if (
        foreign_job
        or foreign_memory
        or foreign_link
        or conflicting_chat
        or any(j.user_id != book.user_id for j in jobs)
    ):
        raise HTTPException(
            409,
            "Another record refers to this book’s storage. Resolve that link before deleting it.",
        )

    paths = set()
    for run_id in run_ids:
        candidates = {
            _target(root, run_id)
            for root in (RUNS_DIR, BASE_DIR / "runs", BASE_DIR.parent / "runs")
        }
        existing = {path for path in candidates if path.exists()}
        if len(existing) > 1:
            raise HTTPException(
                409,
                "This run has multiple storage locations. Resolve the duplicate before deleting it.",
            )
        paths.update(existing)
    vector_root = BASE_DIR.parent / Path(Config(require_api_key=False).vector_db_path)
    for key in keys:
        paths.add(_target(vector_root / "memory", "run_" + key))
    # Keep ownership metadata until every directory has been cleared. A retry
    # after a restart can safely finish without guessing which files belonged here.
    book.deleting = True
    db.commit()
    for path in paths:
        if path.exists():
            try:
                shutil.rmtree(path)
            except OSError as exc:
                raise HTTPException(
                    503,
                    "Some files could not be removed. Close any open files and retry deletion from Books.",
                ) from exc
    db.execute(delete(IdeaCopilotSession).where(IdeaCopilotSession.book_id == book.id))
    db.execute(delete(GenerationJob).where(GenerationJob.book_id == book.id))
    db.add(RetiredBookId(id=book.id))
    db.delete(book)
    db.commit()
