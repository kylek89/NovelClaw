"""Keep each writing tab, resource and generation bound to a durable book."""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from uuid import uuid4

from fastapi import Depends, HTTPException, Request
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from .db import get_db
from .book_activity import book_activity
from .models import Book, RetiredBookId, GenerationJob, IdeaCopilotSession
from .settings import settings

current_book = ContextVar("novelclaw_book", default=None)
WORKSPACE_ROOTS = {
    "dashboard",
    "console",
    "idea-copilot",
    "jobs",
    "providers",
    "api-keys",
    "capabilities",
}


def workspace_path(path):
    return (
        path.split("?", 1)[0].strip("/").split("/", 1)[0] in WORKSPACE_ROOTS
        or path == "/api/idea-copilot"
        or path.startswith(("/api/runs/", "/api/idea-copilot/"))
    )


def new_book(db, user_id, title, legacy_run=""):
    book = Book(
        user_id=user_id,
        title=title[:160],
        memory_key=legacy_run or "book_" + uuid4().hex,
    )
    db.add(book)
    db.flush()
    last_retired = db.scalar(select(func.max(RetiredBookId.id))) or 0
    if book.id <= last_retired:
        book.id = max(last_retired, db.scalar(select(func.max(Book.id)))) + 1
        db.flush()
    return book


def upgrade_books(engine):
    # create_all leaves existing tables intact. Add nullable ownership first.
    with engine.begin() as conn:
        if "kind" not in {c["name"] for c in inspect(conn).get_columns("provider_configs")}:
            conn.execute(text("ALTER TABLE provider_configs ADD COLUMN kind VARCHAR(16) DEFAULT 'custom' NOT NULL"))
        if "source_message" not in {c["name"] for c in inspect(conn).get_columns("book_document_revisions")}:
            conn.execute(text("ALTER TABLE book_document_revisions ADD COLUMN source_message INTEGER"))
        book_columns = {c["name"] for c in inspect(conn).get_columns("books")}
        if "series" not in book_columns:
            conn.execute(
                text(
                    "ALTER TABLE books ADD COLUMN series VARCHAR(160) DEFAULT '' NOT NULL"
                )
            )
        if "deleting" not in book_columns:
            conn.execute(
                text("ALTER TABLE books ADD COLUMN deleting BOOLEAN DEFAULT 0 NOT NULL")
            )
        for table in ("generation_jobs", "idea_copilot_sessions"):
            columns = {c["name"] for c in inspect(conn).get_columns(table)}
            if table == "generation_jobs" and "document_versions" not in columns:
                conn.execute(text("ALTER TABLE generation_jobs ADD COLUMN document_versions TEXT"))
            if table == "generation_jobs" and "document_chars" not in columns:
                conn.execute(text("ALTER TABLE generation_jobs ADD COLUMN document_chars INTEGER"))
            if "model_selection" not in columns:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN model_selection TEXT"))
            if "book_id" not in columns:
                conn.execute(
                    text(
                        f"ALTER TABLE {table} ADD COLUMN book_id INTEGER REFERENCES books(id)"
                    )
                )
                conn.execute(
                    text(f"CREATE INDEX ix_{table}_book_id ON {table} (book_id)")
                )
    with Session(engine) as db, db.begin():
        for session in db.scalars(
            select(IdeaCopilotSession).where(IdeaCopilotSession.book_id.is_(None))
        ):
            job = (
                db.get(GenerationJob, session.final_job_id)
                if session.final_job_id
                else None
            )
            if job and job.user_id != session.user_id:
                job = None
            if job and job.book_id:
                session.book_id = job.book_id
            else:
                book = new_book(
                    db,
                    session.user_id,
                    session.refined_idea or session.original_idea or "Untitled book",
                    job.run_id if job else "",
                )
                session.book_id = book.id
                if job:
                    job.book_id = book.id
        db.flush()
        for job in db.scalars(
            select(GenerationJob).where(GenerationJob.book_id.is_(None))
        ):
            job.book_id = new_book(
                db, job.user_id, job.idea or "Untitled book", job.run_id
            ).id


def child_for_request(request, db, user_id):
    args = request.path_params
    if "run_id" in args:
        return db.scalar(
            select(GenerationJob).where(
                GenerationJob.user_id == user_id, GenerationJob.run_id == args["run_id"]
            )
        )
    if "job_id" in args:
        return db.get(GenerationJob, args["job_id"])
    session_id = args.get("session_id") or request.query_params.get("session_id")
    if session_id and str(session_id).isdigit():
        return db.get(IdeaCopilotSession, int(session_id))
    return None


async def bind_book(request: Request, db: Session = Depends(get_db)):
    book_id = request.path_params.get("book_id")
    writing = request.method not in {"GET", "HEAD", "OPTIONS"}
    if writing and "story_id" in request.path_params:
        job = db.get(GenerationJob, request.path_params["story_id"])
        book_id = job.book_id if job else None
    deleting = request.url.path.endswith("/delete-book")
    with book_activity(book_id if writing and not deleting else None):
        async with _bind_workspace(request, db):
            yield


@asynccontextmanager
async def _bind_workspace(request: Request, db: Session):
    args = request.path_params
    book_id = args.get("book_id")
    path = request.url.path
    console_path = path.partition("/console/")[2]
    request.state.switch_tool = (
        "/console/" + console_path if console_path else "/console/chat"
    )
    scoped = book_id is not None
    book = None
    if scoped or workspace_path(path):
        uid = request.session.get("uid")
        owned = select(Book).where(Book.user_id == uid) if uid else None
        if scoped:
            book = db.scalar(owned.where(Book.id == book_id)) if uid else None
            if book is None or (book.deleting and not path.endswith("/delete-book")):
                raise HTTPException(404, "Book not found")
        resource = any(k in args for k in ("job_id", "session_id", "run_id")) or bool(
            request.query_params.get("session_id")
        )
        memory_view = (
            scoped
            and args.get("run_id") == "book-memory"
            and ("/memory-banks" in path or "/outlines/" in path)
        )
        if uid and resource and not memory_view:
            child = child_for_request(request, db, uid)
            if (
                not child
                or child.user_id != uid
                or (scoped and child.book_id != book.id)
            ):
                raise HTTPException(404, "Item not found in this book")
            if not scoped and child.book_id:
                target = f"{settings.base_path}/books/{child.book_id}{path}"
                if request.url.query:
                    target += "?" + request.url.query
                raise HTTPException(307, headers={"Location": target})
        elif not scoped and uid and not resource:
            available = owned.where(Book.deleting.is_(False))
            target = db.scalar(
                available.where(Book.id == request.session.get("last_book"))
            )
            if target is None:
                target = db.scalar(available.order_by(Book.id).limit(1))
            if target:
                if request.method == "GET":
                    location = f"{settings.base_path}/books/{target.id}{path}"
                    if request.url.query:
                        location += "?" + request.url.query
                    raise HTTPException(307, headers={"Location": location})
                raise HTTPException(409, "Open this action from the selected book.")
        if (
            scoped
            and request.method == "GET"
            and ("/console/" in path or path.endswith("/dashboard"))
        ):
            request.session["last_book"] = book.id
    request.state.book = book
    token = current_book.set(book)
    try:
        yield
    finally:
        current_book.reset(token)
