"""Keep each writing tab, resource and generation bound to a durable book."""

from contextvars import ContextVar
from uuid import uuid4

from fastapi import Depends, HTTPException, Request
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from .db import get_db
from .models import Book, GenerationJob, IdeaCopilotSession
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
    return book


def upgrade_books(engine):
    # create_all leaves existing tables intact. Add nullable ownership first.
    with engine.begin() as conn:
        for table in ("generation_jobs", "idea_copilot_sessions"):
            if "book_id" not in {c["name"] for c in inspect(conn).get_columns(table)}:
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
    args = request.path_params
    book_id = args.get("book_id")
    path = request.url.path
    console_path = path.partition("/console/")[2]
    request.state.switch_tool = (
        "/console/" + console_path if console_path else "/console/chat"
    )
    scoped = book_id is not None
    book = None
    request.state.books = []
    if scoped or workspace_path(path):
        uid = request.session.get("uid")
        if uid:
            request.state.books = list(
                db.scalars(select(Book).where(Book.user_id == uid).order_by(Book.id))
            )
        if scoped:
            book = next(
                (item for item in request.state.books if item.id == book_id), None
            )
            if book is None:
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
        elif not scoped and request.state.books and not resource:
            preferred = request.session.get("last_book")
            target = next(
                (b for b in request.state.books if b.id == preferred),
                request.state.books[0],
            )
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
