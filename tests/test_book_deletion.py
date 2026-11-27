import importlib
import re
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import select


def test_permanent_deletion_is_confirmed_scoped_and_retryable(signed_in, monkeypatch):
    a = signed_in
    if a.name != "novelclaw":
        pytest.skip("Claw workspace only")
    catalog = importlib.import_module("local_web_portal.app.book_library")
    monkeypatch.setattr(catalog, "BASE_DIR", a.temp / "service" / "web")
    user = a.user_id
    ids, files = [], []
    for index in range(2):
        a.client.post("/books", data={"title": "Same title"}, follow_redirects=False)
        with a.db.SessionLocal() as db:
            book = db.scalar(
                select(a.models.Book)
                .where(a.models.Book.user_id == user)
                .order_by(a.models.Book.id.desc())
            )
            run_id = "delete_test_" + uuid4().hex
            job = a.models.GenerationJob(
                book_id=book.id,
                user_id=user,
                idea="sample",
                provider="fixture",
                status="succeeded",
                run_id=run_id,
            )
            db.add(job)
            db.flush()
            db.add(
                a.models.IdeaCopilotSession(
                    user_id=user,
                    book_id=book.id,
                    original_idea="private chat",
                    provider="fixture",
                    final_job_id=job.id,
                )
            )
            db.commit()
            ids.append((book.id, job.id))
            chapter = catalog.RUNS_DIR / run_id / "chapters" / "one.txt"
            memory = (
                a.temp
                / "vector_db"
                / "memory"
                / ("run_" + book.memory_key)
                / "memory_index.json"
            )
            for dest in (chapter, memory):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f"Book {index} canary".encode())
            files.append((chapter, memory))
    # Delete the highest ID to exercise SQLite's normal row-ID reuse case.
    target, job_id = ids[1]
    keep = ids[0][0]
    url = f"/books/{target}/delete-book"
    view = a.client.get(url)
    token = re.search(r'name="confirmation" value="([^"]+)"', view.text)[1]
    assert a.client.post(url, data={"confirmation": "invalid"}).status_code == 409
    with a.db.SessionLocal() as db:
        job = db.get(a.models.GenerationJob, job_id)
        job.status = "running"
        db.commit()
    assert a.client.post(url, data={"confirmation": token}).status_code == 409
    assert all(p.is_file() for p in files[1])
    with a.db.SessionLocal() as db:
        db.get(a.models.GenerationJob, job_id).status = "succeeded"
        db.commit()
    # Simulate an open file, then finish using the same confirmation after retry.
    with patch.object(catalog.shutil, "rmtree", side_effect=PermissionError):
        assert a.client.post(url, data={"confirmation": token}).status_code == 503
    assert a.client.get(f"/books/{target}/console/chat").status_code == 404
    assert (
        a.client.post(
            url, data={"confirmation": token}, follow_redirects=False
        ).status_code
        == 303
    )
    with a.db.SessionLocal() as db:
        assert db.get(a.models.Book, target) is None
        assert db.get(a.models.GenerationJob, job_id) is None
        assert not db.scalar(
            select(a.models.IdeaCopilotSession.id).where(
                a.models.IdeaCopilotSession.book_id == target
            )
        )
        assert db.get(a.models.Book, keep).title == "Same title"
        assert db.get(a.models.GenerationJob, ids[0][1]).status == "succeeded"
        assert (
            db.scalar(
                select(a.models.IdeaCopilotSession.original_idea).where(
                    a.models.IdeaCopilotSession.book_id == keep
                )
            )
            == "private chat"
        )
    assert all(not p.parent.exists() for p in files[1])
    assert all(p.read_bytes() == b"Book 0 canary" for p in files[0])
    created = a.client.post(
        "/books", data={"title": "Replacement"}, follow_redirects=False
    )
    assert created.headers["location"] != f"/books/{target}/console/chat"
    assert a.client.get(url).status_code == 404
