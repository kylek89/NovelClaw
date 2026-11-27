import json
from uuid import uuid4

import pytest


@pytest.fixture
def book_workspace(signed_in):
    if signed_in.name != "novelclaw":
        pytest.skip("Author Books belong to NovelClaw")
    return signed_in


def test_portable_two_chat_writing_journey(book_workspace, monkeypatch):
    ws = book_workspace
    client = ws.client
    assert client.get("/books").status_code == 200
    row = client.post(
        "/api/books",
        json={"title": "Ferry rehearsal", "premise": "A ferryman needs a partner."},
    ).json()
    url = "/api/books/" + row["id"]

    def snapshot():
        return client.get(url + "/backup").json()["book"]

    chat = snapshot()["chats"][0]["id"]
    version = 1

    def edit(action, **other):
        nonlocal version
        r = client.post(
            url + "/edit", json={"revision": version, "action": action, **other}
        )
        assert r.is_success, r.text
        version += 1
        return r.json()

    prompts = []
    outputs = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-local-only")

    def generate(self, messages, **kwargs):
        prompts.append(json.dumps(messages))
        return outputs.pop(0)

    monkeypatch.setattr(ws.main.LLMClient, "chat", generate)

    def ask(mode, output, thread=chat, **extra):
        nonlocal version
        token = uuid4().hex
        request = {
            "revision": version,
            "chat": thread,
            "mode": mode,
            "text": "Follow the authored material.",
            "provider": "deepseek",
            "token": token,
            **extra,
        }
        outputs.append(output)
        r = client.post(url + "/ask", json=request)
        assert r.is_success, r.text
        assert (
            client.post(url + "/ask", json=request).json()["status"]
            == r.json()["status"]
        )
        version += 2
        return r.json()

    ask("discover", '[{"name":"Keeper","profile":"Works at the ferry."}]')
    msg = snapshot()["chats"][0]["messages"][-1]
    edit(
        "save_candidates",
        chat=chat,
        message=msg["id"],
        picks=[{"index": 0, "name": "Mira", "profile": "Keeps the keys."}],
    )
    char = snapshot()["cast"][0]["id"]
    thread = edit("develop", character=char)["chat"]
    ask("chat", "Restless at low tide.\n\nShe keeps a brass key.", thread)
    profile_msg = snapshot()["chats"][1]["messages"][-1]
    edit("profile_capture", chat=thread, message=profile_msg["id"], selected=[1])
    assert (
        snapshot()["cast"][0]["profile"] == "Keeps the keys.\n\nShe keeps a brass key."
    )
    edit(
        "character",
        character=char,
        name="Mira",
        text="Keeps the keys.",
        private="PRIVATE_CANARY",
    )
    outlines = [{"chapter": n, "outline": f"Crossing {n}."} for n in range(1, 6)]
    ask("batch", json.dumps(outlines))
    saved = snapshot()
    assert len(saved["chapters"]) == 5
    ch = saved["chapters"][0]["id"]
    sc = saved["chapters"][0]["scenes"][0]["id"]
    ask("prose", "Mira keeps watch.", chapter=ch, scene=sc)
    scene2 = edit(
        "split",
        chapter=ch,
        scene=sc,
        text="Mira keeps watch.\nThe ferry arrives.",
        offset=17,
        name="Watch",
        time="Day 1 morning",
        pov="Mira",
    )["scene"]
    edit(
        "scene",
        chapter=ch,
        scene=scene2,
        name="Arrival",
        text="The ferry arrives.",
        pov="Mira",
        time="Day 1 night",
        direction="Ring the bell.",
    )
    edit(
        "fact",
        character=char,
        chapter=ch,
        scene=scene2,
        text="NOW_CAPTAIN",
        tag="Relationship",
        excerpt="The ferry arrives.",
    )
    ask("prose", "She waits.", chapter=ch, scene=sc)
    assert "NOW_CAPTAIN" not in prompts[-1] and "PRIVATE_CANARY" not in prompts[-1]
    ask("prose", "The bell rings.", chapter=ch, scene=scene2)
    assert (
        "NOW_CAPTAIN" in prompts[-1]
        and "Ring the bell." in prompts[-1]
        and "PRIVATE_CANARY" not in prompts[-1]
    )
    assert (
        snapshot()["chapters"][0]["scenes"][1]["prose"]
        == "The ferry arrives.\n\nThe bell rings."
    )
    assert client.get("/books/" + row["id"] + "?view=cast").status_code == 200
    assert (
        client.get("/books/" + row["id"] + "?view=chat&chat=" + thread).status_code
        == 200
    )
    assert client.get("/books/" + row["id"] + "?view=prose").status_code == 200
    late = client.post(
        url + "/edit",
        json={"revision": 1, "action": "premise", "name": "Wrong", "text": "Wrong"},
    )
    assert late.status_code == 409
    package = client.get(url + "/backup").json()
    restored = client.post("/api/books/restore", json=package).json()
    assert restored["id"] != row["id"]
    assert client.get("/api/books/" + restored["id"] + "/backup").json() == package
    assert len(prompts) == 6
    moved = edit("review_next", chapter=ch, scene=sc)
    assert (
        moved["scene"] == scene2 and snapshot()["chapters"][0]["scenes"][0]["reviewed"]
    )
    edit(
        "scene",
        chapter=ch,
        scene=scene2,
        name="Arrival",
        text="The ferry has moved.",
        time="Day 2",
        pov="Mira",
    )
    ask("prose", "Mira follows.", chapter=ch, scene=scene2)
    assert "NOW_CAPTAIN" not in prompts[-1]
    assert "Source edited" in client.get("/books/" + row["id"] + "?view=cast").text
    assert len(prompts) == 7


def test_full_legacy_storyboard_and_cast(book_workspace):
    main = book_workspace.main
    entities = [
        {
            "name": f"Keeper {i}",
            "content": "Role: Ferryman",
            "attributes": {"role": "Ferryman"},
        }
        for i in range(15)
    ]
    records = [
        {
            "id": str(n),
            "content": f"Crossing {n}",
            "structure": {"chapter": n, "kind": "chapter_outline"},
        }
        for n in range(1, 41)
    ]
    assets = main._build_story_assets(
        {
            "index": {
                "outlines": records,
                "plot_points": records,
                "characters": entities,
            }
        },
        None,
        None,
    )
    board = main._build_storyboard_view(assets, "en")
    assert len(board["outlines"]) == len(board["plot_points"]) == 40
    assert len(main._build_character_cards(assets, "en")) == 15


def test_pov_knowledge_and_private_context(book_workspace):
    from local_web_portal.app.book_data import (
        BookData,
        Chapter,
        Character,
        Fact,
        Scene,
        writing_context,
    )

    scene1 = Scene(pov="Mira", prose="The lamp is broken.")
    scene2 = Scene(pov="Omar")
    chapter = Chapter(title="Crossing", scenes=[scene1, scene2])
    mira = Character(
        name="Mira",
        private="PRIVATE_PROFILE",
        history=[
            Fact(scene=scene1.id, tag="Knowledge", value="SECRET_KEY", excerpt="lamp"),
            Fact(scene=scene1.id, tag="NSFW", value="PRIVATE_FACT", excerpt="lamp"),
        ],
    )
    book = BookData(title="Ferry", chapters=[chapter], cast=[mira])

    def context(lane):
        return writing_context(book, book.chats[0], lane, chapter, scene2)

    assert "SECRET_KEY" not in context("sfw") and "PRIVATE_FACT" not in context("sfw")
    assert "lamp is broken" not in context("sfw")
    scene2.pov = "Mira"
    assert "SECRET_KEY" in context("sfw") and "PRIVATE_PROFILE" not in context("sfw")
    assert "PRIVATE_PROFILE" in context("nsfw") and "PRIVATE_FACT" in context("nsfw")


def test_generation_preserves_edits_and_incomplete_results(book_workspace, monkeypatch):
    client = book_workspace.client
    row = client.post(
        "/api/books",
        json={
            "title": "Night crossing",
            "chapters": [{"title": "Pier", "outline": "Original."}],
        },
    ).json()
    url = "/api/books/" + row["id"]

    def snapshot():
        return client.get(url + "/backup").json()["book"]

    saved = snapshot()
    req = dict(
        chat=saved["chats"][0]["id"],
        provider="deepseek",
        text="Write the outline.",
        mode="outline",
        chapter=saved["chapters"][0]["id"],
        revision=1,
        token=uuid4().hex,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-test")
    assert client.post(url + "/ask", json=req).status_code == 409

    def delayed(self, messages, **kwargs):
        r = client.post(
            url + "/edit",
            json={
                "action": "outline",
                "revision": 2,
                "chapter": req["chapter"],
                "text": "Author edited while waiting.",
            },
        )
        assert r.is_success, r.text
        return "Claw's new draft."

    monkeypatch.setattr(book_workspace.main.LLMClient, "chat", delayed)
    assert (
        client.post(url + "/ask", json={**req, "replace": True}).json()["status"]
        == "chat_only"
    )
    assert snapshot()["chapters"][0]["outline"] == "Author edited while waiting."
    assert snapshot()["chats"][0]["messages"][-1]["text"] == "Claw's new draft."
    monkeypatch.setattr(
        book_workspace.main.LLMClient,
        "chat",
        lambda *args, **kw: '[{"chapter":2,"outline":"Incomplete"}]',
    )
    r = client.post(
        url + "/ask",
        json={
            **req,
            "token": uuid4().hex,
            "revision": 4,
            "mode": "batch",
            "first": 2,
            "last": 4,
        },
    )
    assert r.json()["status"] == "format_error"
    assert len(snapshot()["chapters"]) == 1
