"""Ownership, compatibility and isolation checks for the Claw book picker."""

import importlib
import json
from urllib.parse import urlsplit

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


def test_migrate_legacy_library(workspace, tmp_path):
    if workspace.name != "novelclaw":
        pytest.skip("Claw workspace only")
    db_engine = create_engine(f"sqlite:///{(tmp_path / 'prior.db').as_posix()}")
    m = workspace.models
    m.Base.metadata.create_all(db_engine)
    with Session(db_engine) as db, db.begin():
        user = m.User(email="prior@example.test", password_hash="fixture")
        db.add(user)
        db.flush()
        job = m.GenerationJob(
            user_id=user.id,
            provider="fixture",
            status="succeeded",
            idea="original prose",
            run_id="prior_run",
        )
        db.add(job)
        db.flush()
        for link in (job.id, None):
            db.add(
                m.IdeaCopilotSession(
                    user_id=user.id,
                    provider="fixture",
                    original_idea="Identical title",
                    conversation_json='{"old":"chat"}',
                    final_job_id=link,
                )
            )
        db.add(
            m.GenerationJob(
                user_id=user.id,
                provider="fixture",
                idea="Standalone run",
                run_id="orphan_run",
            )
        )
    with db_engine.begin() as conn:
        for table in ("generation_jobs", "idea_copilot_sessions"):
            # Rebuild the fixture with its original columns, including the PK.
            columns = [
                r[1]
                for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")
                if r[1] != "book_id"
            ]
            fields = ", ".join(columns)
            ddl = conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).scalar_one()
            legacy = "\n".join(
                line for line in ddl.splitlines() if "book_id" not in line
            )
            conn.exec_driver_sql(
                legacy.replace("CREATE TABLE " + table, "CREATE TABLE prior_" + table)
            )
            conn.exec_driver_sql(
                f"INSERT INTO prior_{table} ({fields}) SELECT {fields} FROM {table}"
            )
            conn.exec_driver_sql(f"DROP TABLE {table}")
            conn.exec_driver_sql(f"ALTER TABLE prior_{table} RENAME TO {table}")
        conn.exec_driver_sql("DROP TABLE books")
    m.Base.metadata.create_all(db_engine)
    books = importlib.import_module("local_web_portal.app.book_workspace")
    books.upgrade_books(db_engine)
    books.upgrade_books(db_engine)
    with Session(db_engine) as db:
        assert len(list(db.scalars(select(m.Book)))) == 3
        sessions = list(
            db.scalars(select(m.IdeaCopilotSession).order_by(m.IdeaCopilotSession.id))
        )
        assert sessions[0].book_id != sessions[1].book_id
        assert all(s.conversation_json == '{"old":"chat"}' for s in sessions)
        job = db.get(m.GenerationJob, sessions[0].final_job_id)
        assert job.book_id == sessions[0].book_id
        assert db.get(m.Book, job.book_id).memory_key == job.run_id == "prior_run"
        orphan = db.scalar(
            select(m.GenerationJob).where(m.GenerationJob.run_id == "orphan_run")
        )
        assert orphan.book_id not in {s.book_id for s in sessions}
    db_engine.dispose()


def test_old_bookmark_and_new_conversation(signed_in):
    a = signed_in
    if a.name != "novelclaw":
        pytest.skip("Claw workspace only")
    paths = []
    for _ in range(2):
        resp = a.client.post(
            "/books", data={"title": "Same title"}, follow_redirects=False
        )
        assert resp.status_code == 303
        paths.append(
            urlsplit(resp.headers["location"]).path.removesuffix("/console/chat")
        )
    ids = [int(p.split("/")[-1]) for p in paths]
    with a.db.SessionLocal() as db:
        book = db.get(a.models.Book, ids[0])
        run = "old_bookmarked_run"
        job = a.models.GenerationJob(
            user_id=a.user_id,
            book_id=book.id,
            provider="local",
            idea="original draft",
            status="succeeded",
            run_id=run,
        )
        db.add(job)
        db.flush()
        old = a.models.IdeaCopilotSession(
            user_id=a.user_id,
            book_id=book.id,
            provider="local",
            original_idea="original chat",
            final_job_id=job.id,
        )
        db.add(old)
        db.flush()
        sid = old.id
        for i in range(21):
            db.add(
                a.models.IdeaCopilotSession(
                    user_id=a.user_id,
                    book_id=book.id,
                    provider="local",
                    original_idea=str(i),
                )
            )
            db.add(
                a.models.GenerationJob(
                    user_id=a.user_id,
                    book_id=book.id,
                    provider="local",
                    status="draft",
                    idea=str(i),
                )
            )
        db.commit()
    resp = a.client.get(f"/console/manuscript/read?session_id={sid}")
    assert resp.status_code == 200
    assert urlsplit(str(resp.url)).path.startswith(paths[0] + "/")
    assert resp.context["chat_session"].id == sid
    assert resp.context["workspace_bootstrap"]["run_id"] == run
    assert a.client.get(paths[1] + "/console/chat").context["idea_sessions"] == []
    assert (
        a.client.get(paths[0] + "/console/chat?new=1").context["chat_session"] is None
    )
    assert a.client.get(paths[1] + f"/console/chat?session_id={sid}").status_code == 404
    empty = a.client.get(paths[1] + "/console/memory/zone/author_brief")
    assert empty.context["workspace_bootstrap"]["memory_target"] == "book-memory"
    assert "original chat" not in json.dumps(empty.context["story_assets"])
