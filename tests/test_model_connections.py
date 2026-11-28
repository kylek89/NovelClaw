"""New selections and legacy records, with optional real Lenovo acceptance."""

import json
import os
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from sqlalchemy import select


@pytest.fixture
def author(signed_in):
    if signed_in.name != "novelclaw":
        pytest.skip("NovelClaw provider selection")
    return signed_in


def new_book(a):
    reply = a.client.post(
        "/books", data={"title": "Provider qualification"}, follow_redirects=False
    )
    return urlsplit(reply.headers["location"]).path.removesuffix("/console/chat")


def test_model_pinning_and_local_credentials(author, monkeypatch):
    a = author
    from local_web_portal.app import model_connections as con
    from local_web_portal.app import job_runner
    from local_web_portal.app.provider_registry import ProviderSpec

    book = new_book(a)
    monkeypatch.setenv("LLM_API_KEY", "generic-cloud-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PGX_API_KEY", raising=False)
    with a.db.SessionLocal() as db:
        specs = con.connections(db, a.user_id)
        assert con.credential(db, a.user_id, specs["pgx"]) == "local-no-auth"
        with pytest.raises(ValueError):
            con.credential(db, a.user_id, specs["openrouter"])
        job = a.main._create_generation_job(
            db,
            a.user_id,
            "pgx",
            "Draft",
            book_id=int(book.split("/")[-1]),
            model="pinned-model",
        )
        jid = job.id
    monkeypatch.setenv("PGX_MODEL", "changed-default")
    captured = {}

    def worker(*args, **kwargs):
        captured.update(kwargs)
        assert args[2] == "local-no-auth"
        return {"ok": False, "error": "fixture ended"}

    monkeypatch.setattr(job_runner, "_run_worker_subprocess", worker)
    job_runner.run_generation_job(jid)
    assert captured["provider_specs"]["pgx"].model == "pinned-model"
    with a.db.SessionLocal() as db:
        pinned = db.get(a.models.GenerationJob, jid).model_selection
        with pytest.raises(ValueError, match="address changed"):
            con.select_model(
                ProviderSpec("pgx", "Changed", "http://elsewhere/v1", "other"),
                snapshot=pinned,
            )
    # Manual selection is retained even if the catalog temporarily disappears.
    from provider_server import provider_server, completion

    with provider_server(
        lambda *_: (
            200,
            completion(
                '{"analysis":"Discuss the lighthouse.","questions":["Who lives there?"],"question_options":[["Keeper","Sailor","Writer"]],"readiness":40}'
            ),
        )
    ) as (url, requests):
        monkeypatch.setenv("PGX_BASE_URL", url)
        start = a.client.post(
            book + "/idea-copilot/start",
            data={"provider": "pgx", "model": "first", "idea": "A lighthouse."},
            follow_redirects=False,
        )
        assert start.status_code == 303
        with a.db.SessionLocal() as db:
            chat = db.scalar(
                select(a.models.IdeaCopilotSession).where(
                    a.models.IdeaCopilotSession.user_id == a.user_id
                )
            )
            assert chat.selected_model == "first"
            sid = chat.id
        monkeypatch.setattr(a.main, "_start_idea_copilot_reply_worker", lambda _: None)
        reply = a.client.post(
            book + f"/idea-copilot/{sid}/reply",
            data={"reply": "Make a list.", "provider": "pgx", "model": "second"},
            headers={"X-Requested-With": "fetch"},
        )
        assert reply.status_code == 202
        a.main._run_idea_copilot_reply_worker(sid)
        assert requests[0]["body"]["model"] == "first"
        assert all(r["body"]["model"] == "second" for r in requests[1:])
        with a.db.SessionLocal() as db:
            rows = db.scalars(
                select(a.models.ModelCall).where(
                    a.models.ModelCall.book_id == int(book.split("/")[-1])
                )
            ).all()
            assert {r.requested for r in rows} == {"first", "second"}
            assert all(r.reported == "test-model" for r in rows)
        assert "Book model activity" in a.client.get(book + "/console/models").text


def test_openrouter_catalog_price_model_and_receipt(author, monkeypatch):
    from dataclasses import replace
    from provider_server import provider_server, completion
    from local_web_portal.app import model_connections as con

    a = author
    book = new_book(a)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def respond(*_):
        value = completion(
            '{"analysis":"A keeper returns.","questions":["At dawn?"],"question_options":[["Sunrise","Noon","Dusk"]],"readiness":40}'
        )
        value.update(
            model="vendor/resolved",
            provider="fixture-upstream",
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "cost": 0.00005,
            },
        )
        return 200, value

    catalog = [
        {
            "id": "vendor/test",
            "context_length": 8192,
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0.000001", "completion": "0.000003"},
        }
    ]
    with provider_server(respond, models=catalog) as (url, requests):
        registry = con.get_provider_specs

        def specs(settings):
            out = registry(settings)
            out["openrouter"] = replace(out["openrouter"], base_url=url)
            return out

        monkeypatch.setattr(con, "get_provider_specs", specs)
        rows = a.client.get(book + "/providers/openrouter/models").json()["models"]
        assert rows == [
            {
                "id": "vendor/test",
                "context": 8192,
                "input_price": "1",
                "output_price": "3",
                "tools": True,
            }
        ]
        form = {
            "provider": "openrouter",
            "model": "vendor/test",
            "idea": "A fictional lighthouse.",
        }
        a.client.post(book + "/idea-copilot/start", data=form)
        assert not requests
        a.client.post(
            book + "/api-keys",
            data={"provider": "openrouter", "api_key": "simulated-key-only"},
        )
        a.client.post(book + "/idea-copilot/start", data=form)
        assert requests[-1]["body"]["model"] == "vendor/test"
        with a.db.SessionLocal() as db:
            row = db.scalar(
                select(a.models.ModelCall).where(
                    a.models.ModelCall.book_id == int(book.split("/")[-1])
                )
            )
            assert (
                row.requested,
                row.reported,
                row.upstream,
                row.input_tokens,
                row.output_tokens,
            ) == ("vendor/test", "vendor/resolved", "fixture-upstream", 20, 10)
            assert float(row.cost_usd) == 0.00005
        other = new_book(a)
        assert "fixture-upstream" not in a.client.get(other + "/console/models").text


def test_offline_local_server_and_agent_selection(author, monkeypatch):
    a = author
    book = new_book(a)
    monkeypatch.setenv("PGX_BASE_URL", "http://127.0.0.1:1/v1")
    failed=a.client.get(book+"/providers/pgx/models")
    assert failed.status_code == 503
    assert "server" in failed.json()["detail"]
    headers={"Authorization":"Bearer test-agent-token"}
    story=a.client.post("/api/v1/stories",headers=headers,json={"provider":"pgx","model":"local-pinned","idea":"An offline draft"})
    assert story.status_code==201
    assert story.json()["model"]=="local-pinned" and story.json()["provider"]=="pgx"
    assert a.client.get("/api/v1/providers/pgx/models").status_code==401
    assert a.client.get("/api/v1/providers/pgx/models",headers=headers).status_code==503
    reserved=a.client.post("/api/v1/providers",headers=headers,json={"slug":"openrouter","base_url":"http://127.0.0.1:1/v1","model":"other"})
    assert reserved.status_code==409


@pytest.mark.skipif(
    os.getenv("NOVELCLAW_TEST_PGX") != "1", reason="Requires the author's Lenovo PGX"
)
def test_pgx_real_scenes(author):
    a = author
    book = new_book(a)
    page = a.client.post(book + "/console/manuscript/new-chapter", data={"after": 0})
    doc = page.context["scene_doc"]
    api = book + f"/api/runs/{page.context['scene_run']}/chapters/1/scenes"
    first = doc["scenes"][0]
    first.update(
        text="Daniel checks the mirror.",
        day=1,
        time="09:00",
        facts=[
            {
                "character": "Daniel",
                "trait": "height",
                "value": "5 feet 11 inches",
                "phase": "entry",
            }
        ],
    )
    doc["scenes"].append(
        {
            **first,
            "id": uuid4().hex,
            "text": "Daniel changes height.",
            "time": "16:00",
            "facts": [
                {
                    "character": "Daniel",
                    "trait": "height",
                    "value": "4 feet 2 inches",
                    "phase": "change",
                }
            ],
        }
    )
    saved = a.client.put(api, json=doc).json()
    for provider in ("pgx", "pgx-prose"):
        listing = a.client.get(book + f"/providers/{provider}/models")
        model = listing.json()["models"][0]["id"]
        result = a.client.post(
            api + "/" + first["id"] + "/ask",
            data={
                "provider": provider,
                "model": model,
                "revision": saved["revision"],
                "question": "State Daniel's current height in words. Only the height.",
            },
        )
        assert result.status_code == 200, result.text
        reply = result.json()["reply"].lower()
        assert ("11" in reply or "eleven" in reply) and "four" not in reply, reply
        print(provider, model, reply)
    with a.db.SessionLocal() as db:
        records = db.scalars(
            select(a.models.ModelCall).where(
                a.models.ModelCall.book_id == int(book.split("/")[-1])
            )
        ).all()
        assert len(records) == 2 and all(
            r.output_tokens and r.reported for r in records
        )


@pytest.mark.skipif(
    os.getenv("NOVELCLAW_TEST_PGX") != "1", reason="Requires the author's Lenovo PGX"
)
def test_pgx_real_chapter(author):
    a = author
    from local_web_portal.app import job_runner

    book = new_book(a)
    with a.db.SessionLocal() as db:
        job = a.main._create_generation_job(
            db,
            a.user_id,
            "pgx",
            "Write just one short chapter, 150 words. Lin returns a brass key to the lighthouse keeper. End there.",
            book_id=int(book.split("/")[-1]),
        )
        jid = job.id
    job_runner.run_generation_job(jid)
    with a.db.SessionLocal() as db:
        job = db.get(a.models.GenerationJob, jid)
        assert job.status == "succeeded", job.error_message[-2500:]
        chapters = a.main._load_chapter_outputs(a.main._resolve_run_dir(job.run_id))
        assert chapters and chapters[0]["content"].strip()
        book_row = db.get(a.models.Book, job.book_id)
        memory = a.main._load_memory_index_for_run(book_row.memory_key)
        assert memory["texts"] and memory["outlines"]
        calls = db.scalars(
            select(a.models.ModelCall).where(a.models.ModelCall.book_id == job.book_id)
        ).all()
        assert calls and all(c.provider == "pgx" for c in calls)
        print(
            "PGX chapter",
            len(chapters),
            "chapters",
            len(calls),
            "requests",
            len(chapters[0]["content"].split()),
            "words",
        )
