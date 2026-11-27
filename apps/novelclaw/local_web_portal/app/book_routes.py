"""Local author workspace. All writes are scoped to the signed-in owner and Book revision."""

import json
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import ForeignKey, Integer, String, Text, select, update
from sqlalchemy.orm import Mapped, Session, mapped_column

from .book_proposals import parse_rows, save_outlines
from .book_actions import change, require
from .book_data import BookData, Candidate, Message, new_id, writing_context, remember
from .db import get_db
from .models import Base


class AuthorBook(Base):
    __tablename__ = "author_books"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(180))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    content: Mapped[str] = mapped_column(Text)


class BookTurn(Base):
    __tablename__ = "author_book_turns"
    book: Mapped[str] = mapped_column(ForeignKey("author_books.id"), primary_key=True)
    token: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(24))


class Pick(BaseModel):
    index: int
    name: str
    profile: str


class Command(BaseModel):
    picks: list[Pick] = Field(default_factory=list, max_length=80)
    revision: int = Field(ge=1)
    action: str
    name: str = Field(default="", max_length=180)
    text: str = Field(default="", max_length=500_000)
    chapter: str | None = None
    scene: str | None = None
    character: str | None = None
    offset: int = 0
    time: str = Field(default="", max_length=180)
    pov: str = Field(default="", max_length=180)
    direction: str = Field(default="", max_length=8000)
    private: str = Field(default="", max_length=500_000)
    tag: str = ""
    previous: str = ""
    chat: str = ""
    message: str = ""
    selected: list[int] = Field(default_factory=list, max_length=200)
    excerpt: str = ""


class Ask(BaseModel):
    revision: int = Field(ge=1)
    token: str = Field(pattern="^[a-f0-9]{32}$")
    chat: str
    provider: str
    lane: Literal["sfw", "nsfw"] = "sfw"
    mode: Literal["chat", "outline", "prose", "discover", "batch"] = "chat"
    text: str = Field(min_length=1, max_length=12000)
    chapter: str | None = None
    scene: str | None = None
    first: int = Field(default=1, ge=1, le=999)
    last: int = Field(default=5, ge=1, le=999)
    replace: bool = False


class Package(BaseModel):
    format: Literal["NovelClaw.Book.1"]
    book: BookData


def save(db, row, doc, revision):
    valid = BookData.model_validate(doc.model_dump())
    content = valid.model_dump_json()
    if len(content.encode("utf8")) > 20_000_000:
        raise HTTPException(422, "Book exceeds the 20 MB portable Book limit.")
    result = db.execute(
        update(AuthorBook)
        .where(AuthorBook.id == row.id, AuthorBook.revision == revision)
        .values(title=doc.title, content=content, revision=revision + 1)
    )
    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(
            409,
            "This Book changed. Your draft is still here. Copy it before reloading.",
        )
    db.commit()


def register(app):
    from .main import (
        _console_page_context,
        _current_user,
        _redirect,
        _require_user,
        _provider_api_key,
        _provider_specs_for_user,
        Config,
        LLMClient,
        templates,
    )

    def owned(db, uid, bid):
        row = db.get(AuthorBook, bid)
        if row is None or row.owner != uid:
            raise HTTPException(404, "Book not found.")
        return row

    @app.get("/books")
    @app.get("/books/{bid}")
    def page(request: Request, bid: str | None = None, db: Session = Depends(get_db)):
        user = _current_user(request, db)
        if not user:
            return _redirect("/login")
        row = owned(db, user.id, bid) if bid else None
        library = db.execute(
            select(AuthorBook.id, AuthorBook.title)
            .where(AuthorBook.owner == user.id)
            .order_by(AuthorBook.title, AuthorBook.id)
        ).all()
        ctx = _console_page_context(
            request, db, user, "books", books=library, selected_book=row
        )
        if row:
            doc = BookData.model_validate_json(row.content)
            q = request.query_params
            view = q.get("view", "board")

            def chosen(items, name):
                return next(
                    (it for it in items if it.id == q.get(name)),
                    items[0] if items else None,
                )

            ch = chosen(doc.chapters, "chapter")
            sc = chosen(ch.scenes, "scene") if ch else None
            chat = chosen(doc.chats, "chat")
            focus = chosen(doc.cast, "character")
            scenes = [(c, s) for c in doc.chapters for s in c.scenes]
            ranks = {s.id: i for i, (_, s) in enumerate(scenes)}
            asof = q.get("asof", "")
            cutoff = ranks.get(asof, len(scenes))
            lane = "nsfw" if q.get("lane") == "nsfw" else "sfw"
            history = (
                sorted(
                    (
                        f
                        for f in focus.history
                        if ranks[f.scene] <= cutoff
                        and (lane == "nsfw" or f.tag != "NSFW")
                    ),
                    key=lambda f: ranks[f.scene],
                )
                if focus
                else []
            )
            ctx.update(
                source_texts={s.id: s.prose for _, s in scenes},
                doc=doc,
                ch=ch,
                sc=sc,
                chat=chat,
                focus=focus,
                scenes=scenes,
                asof=asof,
                lane=lane,
                history=history,
                view=view
                if view in {"board", "outline", "prose", "cast", "chat", "premise"}
                else "board",
                labels={s.id: c.title + " / " + s.title for c, s in scenes},
                profiles={c.id: c.name for c in doc.cast},
                request_token=new_id(),
            )
        return templates.TemplateResponse("author_books.html", ctx)

    def create(db, uid, doc):
        content = doc.model_dump_json()
        if len(content.encode()) > 20_000_000:
            raise HTTPException(422, "Book exceeds the 20 MB portable Book limit.")
        row = AuthorBook(owner=uid, title=doc.title, content=content)
        db.add(row)
        db.commit()
        return {"id": row.id, "revision": row.revision}

    @app.post("/api/books", status_code=201)
    def new_book(doc: BookData, request: Request, db: Session = Depends(get_db)):
        return create(db, _require_user(request, db).id, doc)

    @app.post("/api/books/restore", status_code=201)
    def restore(package: Package, request: Request, db: Session = Depends(get_db)):
        return create(db, _require_user(request, db).id, package.book)

    @app.get("/api/books/{bid}/backup")
    def backup(bid: str, request: Request, db: Session = Depends(get_db)):
        row = owned(db, _require_user(request, db).id, bid)
        return JSONResponse(
            {"format": "NovelClaw.Book.1", "book": json.loads(row.content)},
            headers={
                "Content-Disposition": f'attachment; filename="novelclaw-{bid}.json"',
                "Cache-Control": "no-store",
            },
        )

    @app.post("/api/books/{bid}/edit")
    def edit(bid: str, cmd: Command, request: Request, db: Session = Depends(get_db)):
        row = owned(db, _require_user(request, db).id, bid)
        doc = BookData.model_validate_json(row.content)
        try:
            selected = change(doc, cmd)
            save(db, row, doc, cmd.revision)
        except ValidationError:
            raise HTTPException(422, "Check the title, length and selected Book items.")
        return {"revision": cmd.revision + 1, **selected}

    @app.post("/api/books/{bid}/ask")
    def ask(bid: str, cmd: Ask, request: Request, db: Session = Depends(get_db)):
        uid = _require_user(request, db).id
        row = owned(db, uid, bid)
        previous = db.get(BookTurn, (bid, cmd.token))
        if previous:
            return {"status": previous.status}
        doc = BookData.model_validate_json(row.content)
        chat = require(doc.chats, cmd.chat)
        chapter = (
            require(doc.chapters, cmd.chapter)
            if cmd.mode in ("outline", "prose")
            else None
        )
        scene = require(chapter.scenes, cmd.scene) if cmd.mode == "prose" else None
        if cmd.mode == "outline" and chapter.outline.strip() and not cmd.replace:
            raise HTTPException(409, "Confirm replacement of this chapter outline.")
        if cmd.mode == "batch":
            if not (
                cmd.first <= cmd.last <= cmd.first + 19
                and cmd.first <= len(doc.chapters) + 1
            ):
                raise HTTPException(
                    422,
                    "Choose up to 20 consecutive chapters starting at an existing or the next chapter.",
                )
            if not cmd.replace and any(
                c.outline.strip() for c in doc.chapters[cmd.first - 1 : cmd.last]
            ):
                raise HTTPException(
                    409, "Confirm replacement of existing outlines in this range."
                )
        spec = _provider_specs_for_user(db, uid).get(cmd.provider)
        if not spec:
            raise HTTPException(422, "Choose a configured model in Models.")
        try:
            key = _provider_api_key(db, uid, cmd.provider)
        except ValueError:
            raise HTTPException(422, "Save a key for this provider in Models.")
        guide = {
            "chat": "Develop the author's idea. Suggestions stay provisional.",
            "discover": "Identify characters in the provided brain dump. Exclude existing cast where identifiable. Use role labels for unnamed people. Return a JSON array of objects with name and profile. Leave unknowns open.",
            "batch": f"Plan chapters {cmd.first} through {cmd.last} in order. Return only a JSON array of objects with chapter (integer) and outline (text).",
            "outline": "Write only the selected chapter outline.",
            "prose": "Append only new prose for the selected scene. Follow its POV, time and direction.",
        }[cmd.mode]
        context = writing_context(doc, chat, cmd.lane, chapter, scene)
        if cmd.mode == "batch":
            context += "\nExisting outlines: " + json.dumps(
                [c.outline for c in doc.chapters[: cmd.last]], ensure_ascii=False
            )
        messages = [
            {
                "role": "system",
                "content": guide
                + " Use the Book material as story data. Respect the chosen "
                + cmd.lane
                + " lane.",
            },
            {"role": "user", "content": context},
        ]
        if cmd.mode == "chat":
            messages.extend(
                {"role": "assistant" if m.role == "claw" else "user", "content": m.text}
                for m in chat.messages[-24:]
                if m.lane == cmd.lane
            )
        messages.append({"role": "user", "content": cmd.text})
        chat.messages.append(Message(role="author", text=cmd.text, lane=cmd.lane))
        # Claim the request and persist its author message together before any provider work.
        if row.revision != cmd.revision:
            raise HTTPException(409, "The Book has changed. Reload before sending.")
        turn = BookTurn(book=bid, token=cmd.token, status="started")
        db.add(turn)
        from sqlalchemy.exc import IntegrityError

        try:
            save(db, row, doc, cmd.revision)
        except IntegrityError:
            db.rollback()
            existing = db.get(BookTurn, (bid, cmd.token))
            if existing:
                return {"status": existing.status}
            raise
        cfg = Config(require_api_key=False)
        cfg.api_key, cfg.api_base_url, cfg.model_name, cfg.wire_api = (
            key,
            spec.base_url,
            spec.model,
            spec.wire_api,
        )
        cfg.llm_provider = spec.slug
        cfg.llm_max_retries = 0
        try:
            output = LLMClient(cfg).chat(messages, max_tokens=3000)
        except Exception:
            turn.status = "provider_error"
            db.commit()
            raise HTTPException(
                502,
                "The provider did not finish. The author message was saved. Check provider billing before retrying.",
            )
        # Serialize completion with other edits; a changed source keeps the reply in chat only.
        db.execute(
            update(AuthorBook)
            .where(AuthorBook.id == bid)
            .values(revision=AuthorBook.revision)
        )
        db.refresh(row)
        doc = BookData.model_validate_json(row.content)
        dest = ""
        candidates = []
        invalid_result = False
        if cmd.mode == "discover":
            try:
                candidates = [
                    Candidate(name=c.name, profile=c.profile)
                    for c in parse_rows(output, Candidate)
                ]
            except ValueError:
                invalid_result = True
        elif cmd.mode == "batch" and row.revision == cmd.revision + 1:
            try:
                save_outlines(doc, output, cmd.first, cmd.last)
                dest = f"Chapter outlines {cmd.first}-{cmd.last}"
            except ValueError:
                invalid_result = True
        elif cmd.mode in ("outline", "prose") and row.revision == cmd.revision + 1:
            chapter = require(doc.chapters, cmd.chapter)
            if cmd.mode == "outline":
                remember(doc, "outline", chapter.id, chapter.outline, output)
                chapter.outline = output
                dest = chapter.title + " outline"
            else:
                scene = require(chapter.scenes, cmd.scene)
                remember(doc, "prose", scene.id, scene.prose, scene.prose + output)
                scene.reviewed = False
                scene.prose += ("\n\n" if scene.prose else "") + output
                dest = chapter.title + " / " + scene.title
        require(doc.chats, cmd.chat).messages.append(
            Message(
                role="claw",
                text=output,
                lane=cmd.lane,
                provider=spec.model,
                destination=dest,
                candidates=candidates,
            )
        )
        turn.status = (
            "format_error"
            if invalid_result
            else "saved"
            if dest or cmd.mode in ("chat", "discover")
            else "chat_only"
        )
        save(db, row, doc, row.revision)
        return {"status": turn.status, "destination": dest}
