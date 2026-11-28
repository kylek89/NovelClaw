"""Record actual model requests for original scenes, revisions, and chapter jobs."""

import json
from pathlib import Path
import runpy
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from provider_server import completion, provider_server


def test_prose_style_across_the_author_workflow(signed_in, monkeypatch):
    a = signed_in
    if a.name != "novelclaw":
        pytest.skip("Book prose settings belong to Claw Studio")
    from local_web_portal.app import job_runner

    def book(name):
        response = a.client.post("/books", data={"title": name}, follow_redirects=False)
        return urlsplit(response.headers["location"]).path.removesuffix("/console/chat")

    first, second = book("Morning and evening"), book("Different Book")
    style = {
        "level": "explicit",
        "note": "Use measured sentences.",
        "previous_level": "brief",
        "previous_note": "",
    }
    assert a.client.post(first + "/console/style", data=style).status_code == 200
    assert a.client.post(first + "/console/style", data=style).status_code == 409
    assert (
        a.client.get(second + "/console/style").context["prose_settings"].level
        == "brief"
    )
    for name, value in [
        ("Daniel", "PRIVATE DANIEL NOTE"),
        ("Morgan", "PRIVATE MORGAN NOTE"),
    ]:
        assert (
            a.client.post(
                first + "/console/characters",
                data={"name": name, "profile": "Book character"},
            ).status_code
            == 200
        )
        note = {"name": name, "note": value, "previous": ""}
        assert (
            a.client.post(
                first + "/console/characters/intimate-note", data=note
            ).status_code
            == 200
        )
        assert (
            a.client.post(
                first + "/console/characters/intimate-note", data=note
            ).status_code
            == 409
        )
    book_id = int(first.split("/")[-1])
    assert "PRIVATE" not in a.main._book_notes(book_id)
    page = a.client.post(first + "/console/manuscript/new-chapter", data={"after": 0})
    api = first + f"/api/runs/{page.context['scene_run']}/chapters/1/scenes"

    def scene(time, facts, **extra):
        return {
            "id": uuid4().hex,
            "text": "Daniel checks the mirror.",
            "day": 1,
            "time": time,
            "pov": "Daniel",
            "participants": "Daniel",
            "facts": [
                {
                    "character": "Daniel",
                    "trait": trait,
                    "value": value,
                    "phase": phase,
                    "intimate": private,
                }
                for trait, value, phase, private in facts
            ],
            **extra,
        }

    morning = scene(
        "08:00",
        [
            ("height", "5ft11", "entry", False),
            ("relationship", "friends", "entry", False),
            ("personal detail", "PRIVATE FACT", "entry", True),
        ],
        prose_level="closed",
        include_intimate=True,
    )
    change = scene(
        "16:00",
        [
            ("height", "4ft2", "change", False),
            ("relationship", "partners", "change", False),
        ],
    )
    evening = scene("19:00", [], include_intimate=True)
    scenes = [morning, change, evening]
    saved = a.client.put(
        api, json={"revision": page.context["scene_doc"]["revision"], "scenes": scenes}
    )
    assert saved.status_code == 200, saved.text
    revision = saved.json()["revision"]

    checks = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/generation-smoke.py")
    )

    def provider(path, body, count):
        text = json.dumps(body, ensure_ascii=False)
        if "the author's scene writing partner" in text:
            answer = (
                "Preserve the planned POV."
                if "Review the prose" in text and "Draft one." in text
                else "PASS"
            )
            if "Draft this scene." in text:
                answer = "Draft one."
            if "Revise this scene." in text:
                answer = "Revised scene."
            return 200, completion(answer)
        return checks["fixture_reply"](path, body, count)

    def verify(wire, pos):
        assert "POV: Daniel" in wire and "Use measured sentences." in wire
        assert ("5ft11" in wire) == (pos != 2)
        assert ("4ft2" in wire) == (pos != 0)
        assert ("friends" in wire) == (pos != 2)
        assert ("partners" in wire) == (pos != 0)
        assert ("PRIVATE DANIEL NOTE" in wire) == (pos == 2)
        assert ("PRIVATE FACT" in wire) == (pos == 2)
        assert "PRIVATE MORGAN NOTE" not in wire
        assert ("Fade to black" if pos == 0 else "Explicit") in wire

    with provider_server(provider) as (url, requests):
        monkeypatch.setenv("PGX_BASE_URL", url)
        for pos, target in enumerate(scenes):
            endpoint = api + "/" + target["id"]
            ctx = a.client.get(
                endpoint + "/context", params={"revision": revision}
            ).json()["context"]
            verify(ctx, pos)
            for action in ("draft", "review", "revise"):
                response = a.client.post(
                    endpoint + "/ask",
                    data={"action": action, "provider": "pgx", "revision": revision},
                )
                assert response.status_code == 200, response.text
                wire = "\n".join(m["content"] for m in requests[-1]["body"]["messages"])
                assert ctx in wire
                verify(wire, pos)

        requests.clear()
        with a.db.SessionLocal() as db:
            job = a.main._create_generation_job(
                db,
                a.user_id,
                "pgx",
                "Write one short chapter about Daniel.",
                book_id=book_id,
            )
            job_id = job.id
        job_runner.run_generation_job(job_id)
        with a.db.SessionLocal() as db:
            job = db.get(a.models.GenerationJob, job_id)
            assert job.status == "succeeded", job.error_message
            folder = a.main._resolve_run_dir(job.run_id)
            doc = a.main.scene_manuscript.document(
                a.main.latest_chapter_files(folder)[0]
            )
            assert [row["id"] for row in doc["scenes"]] == [row["id"] for row in scenes]
            assert all(row["text"].strip() == "Revised scene." for row in doc["scenes"])
        seen = set()
        for req in requests:
            wire = "\n".join(
                m.get("content") or "" for m in req["body"].get("messages", [])
            )
            if "the author's scene writing partner" not in wire:
                assert "PRIVATE" not in wire
                continue
            pos = next(i for i in range(3) if f"Chapter 1, scene {i + 1}" in wire)
            verify(wire, pos)
            for action in (
                "Draft this scene.",
                "Revise this scene.",
                "Review the prose",
            ):
                if action in wire:
                    seen.add((pos, action))
        assert len(seen) == 9

        # New manuscripts still use the normal chapter agent loop and the Book's style.
        requests.clear()
        assert (
            a.client.post(
                second + "/console/style",
                data={**style, "level": "closed", "note": "Saltwater cadence."},
            ).status_code
            == 200
        )
        with a.db.SessionLocal() as db:
            fresh = a.main._create_generation_job(
                db,
                a.user_id,
                "pgx",
                "Write one short chapter about a lighthouse.",
                book_id=int(second.split("/")[-1]),
            )
            fresh_id = fresh.id
        job_runner.run_generation_job(fresh_id)
        with a.db.SessionLocal() as db:
            fresh = db.get(a.models.GenerationJob, fresh_id)
            assert fresh.status == "succeeded", fresh.error_message
        wires = [json.dumps(req["body"], ensure_ascii=False) for req in requests]
        assert all(
            "PRIVATE" not in wire and "Use measured sentences." not in wire
            for wire in wires
        )
        for task in ("expert long-form fiction writer", "plot consistency checker"):
            assert any(
                task in wire
                and "Fade to black" in wire
                and "Saltwater cadence." in wire
                for wire in wires
            )
    assert (
        a.client.get(
            api + "/" + morning["id"] + "/context", params={"revision": revision}
        ).status_code
        == 200
    )
