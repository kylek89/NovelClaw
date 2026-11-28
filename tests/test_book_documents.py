"""Book documents, conflict recovery, editing history and the writing handoff."""
from urllib.parse import urlsplit

import pytest


def test_revision_workflow_and_pinned_job(signed_in):
    a = signed_in
    if a.name != "novelclaw":
        pytest.skip("Claw Studio Books")

    def create(title):
        response = a.client.post("/books", data={"title": title}, follow_redirects=False)
        return urlsplit(response.headers["location"]).path.removesuffix("/console/chat")

    book = create("Harbor")
    other = create("Forest")
    bid = int(book.split("/")[-1])
    url = book + "/console/memory/zone/premise"
    assert a.client.get(url).context["history"] == []
    def save(**fields):
        return a.client.post(url + "/document", data=fields, follow_redirects=False)
    assert save(version=0, content="Old lighthouse rule").status_code == 303
    assert save(version=0, content="Stale edit").status_code == 409
    assert save(version=1, content="Only the pilot knows").status_code == 303
    assert save(version=2, content="Only the pilot knows").status_code == 303
    assert len(a.client.get(url).context["history"]) == 2
    with a.db.SessionLocal() as db:
        job = a.main._create_generation_job(db, a.user_id, "pgx", "Draft", book_id=bid)
        job_id = job.id
        assert job.idea.count("Only the pilot knows") == 1
        assert "revision 2" in job.idea and "Old lighthouse rule" not in job.idea

    assert save(version=2, restore=1).status_code == 303
    assert a.client.get(url).context["latest"].content == "Old lighthouse rule"
    assert a.client.get(other + "/console/memory/zone/premise").context["latest"] is None
    assert a.main._book_notes(bid).count("Old lighthouse rule") == 1
    assert "Only the pilot knows" not in a.main._book_notes(bid)
    with a.db.SessionLocal() as db:
        assert "Only the pilot knows" in db.get(a.models.GenerationJob, job_id).idea


def test_shorthand_and_wrong_card(signed_in):
    if signed_in.name != "novelclaw":
        pytest.skip("Claw Studio Books")
    main = signed_in.main
    qs = ["Where?", "When?", "Who?"]
    opts = [["Pier", "Bay", "Reef"], ["Dawn", "Noon", "Dusk"], ["Al", "Jo", "Kim"]]
    state = {"messages": [{"role": "assistant", "questions": qs, "question_options": opts}]}
    result = main.append_user_reply(state, "1 - C\r\n2 - A\r\n3 - C")
    assert result["messages"][-1]["selected_options"] == ["Where? — Reef", "When? — Dawn", "Who? — Kim"]
    assert main.resolve_question_answers(state, {"questions": qs, "options": opts, "picks": [None, "Midnight", 0]}) == [
        "When? — Midnight", "Who? — Al"]
    state["refined_idea"] = "The unchosen lighthouse ghost"
    result["refined_idea"] = state["refined_idea"]
    fresh_brief = main.build_generation_idea("A crossing", result, approved_documents={})
    assert all(text in fresh_brief for text in ("A crossing", "Reef", "Dawn", "Kim"))
    assert "ghost" not in fresh_brief and "Midnight" not in fresh_brief
    result["author_edited_summary"] = "An author-edited crossing"
    assert "An author-edited crossing" in main.build_generation_idea("", result, approved_documents={})
    updated = main.build_generation_idea("A crossing", result, approved_documents={"premise": "Where? — Pier"})
    assert "Reef" not in updated and "Dawn" in updated and "ghost" not in updated
    with pytest.raises(ValueError):
        main.resolve_question_answers(state, {"questions": list(reversed(qs)), "options": opts, "picks": [2, None, None]})


def test_long_source_and_chapter_direction_claims(signed_in):
    a = signed_in
    if a.name != "novelclaw":
        pytest.skip("Book workspace")
    m = a.main
    from agents import idea_copilot_agent as agent

    book_url = a.client.post("/books", data={"title": "Source test"}, follow_redirects=False).headers["location"]
    prefix = book_url.split("/console/")[0]
    bid = int(prefix.split("/")[-1])
    full = "Daniel " * 1800 + "last line of the brain dump"
    state = m.append_user_reply({}, full)
    long_reply = "Character detail. " * 220 + "last line of reply"
    turn = agent._normalize_turn({"analysis": long_reply}, "", "", "en")
    for n in range(22):
        state = m.append_assistant_turn(state, turn)
        state = m.append_user_reply(state, f"Follow-up {n}.")
    with a.db.SessionLocal() as db:
        session = a.models.IdeaCopilotSession(user_id=a.user_id, book_id=bid, provider="pgx",
            original_idea="Start", conversation_json=m.dump_state(state))
        db.add(session)
        db.flush()
        sid = session.id
        a.main.book_documents.add_answers(db, bid, sid, {"premise": ["A sealed jar"]}, {"premise": 0}, 1)
        direction = a.models.ChapterDirection(book_id=bid, chapter=2, content="Only in chapter two")
        db.add(direction)
        db.commit()
        first = m._create_generation_job(db, a.user_id, "pgx", "Draft", book_id=bid)
        second = m._create_generation_job(db, a.user_id, "pgx", "Draft", book_id=bid)
        assert direction.job_id == first.id and second.id != first.id
        assert direction.content not in first.idea
        first.status = "failed"
        db.commit()
        retry = m._create_generation_job(db, a.user_id, "pgx", "Retry", book_id=bid)
        assert direction.job_id == retry.id
        direction.used = True
        db.commit()
        m._create_generation_job(db, a.user_id, "pgx", "Again", book_id=bid)
        assert direction.job_id == retry.id
    history = a.client.get(prefix + "/console/memory/zone/premise")
    assert f"?session_id={sid}#message-1" in history.text
    page = a.client.get(prefix + f"/console/chat?session_id={sid}")
    assert full in page.text and long_reply in page.text and 'id="message-45"' in page.text
    assert full not in agent._history_text(state["messages"])


def test_field_review_conflicts_and_outline_revisions(signed_in):
    if signed_in.name != "novelclaw":
        pytest.skip("Book workspace")
    from fastapi import HTTPException
    from local_web_portal.app import craft_answers, book_documents
    data = {"world_settings": [{"id": "pier", "name": "Pier", "content": "Rule: Open daily"}]}
    args = (data, "world", "pier", "Rule", "Open daily", "Closed on Sunday", [1, 3, 0])
    craft_answers.apply(*args)
    craft_answers.apply(*args)
    row = data["world_settings"][0]
    assert len(row["author_changes"]) == 1 and craft_answers.fields(row, "world")["Rule"] == "Closed on Sunday"
    with pytest.raises(HTTPException) as conflict:
        craft_answers.apply(*args[:5], "Open at night", [1, 4, 0])
    assert conflict.value.status_code == 409
    current = {"premise": 2, "author_brief": 1}
    outline = {"structure": {"kind": "chapter_outline", "document_versions": {"premise": 1, "author_brief": 1}}}
    assert book_documents.outline_notice(outline, current).startswith("Story Premise changed")
    outline["structure"]["document_versions"] = current
    assert book_documents.outline_notice(outline, current) == ""


def test_craft_review_from_saved_message(signed_in):
    a = signed_in
    if a.name != "novelclaw":
        pytest.skip("Book workspace")
    url = a.client.post("/books", data={"title": "Cast review"}, follow_redirects=False).headers["location"]
    prefix = url.split("/console/")[0]
    bid = int(prefix.split("/")[-1])
    assert a.client.post(prefix + "/console/characters", data={"name": "Daniel", "profile": "Neighbor"}).status_code == 200
    with a.db.SessionLocal() as db:
        state = a.main.append_user_reply({}, "Daniel repairs watches.")
        state["messages"][0]["scoped_answers"] = [{"kind":"character", "text":"Daniel repairs watches."}]
        session = a.models.IdeaCopilotSession(user_id=a.user_id, book_id=bid, provider="pgx",
            original_idea="Start", conversation_json=a.main.dump_state(state))
        db.add(session)
        db.commit()
        sid = session.id
    url = prefix + f"/console/chat/{sid}/answers/1/0"
    page = a.client.get(url)
    assert page.status_code == 200 and 'id="craft-review"' in page.text
    target, = page.context["targets"]
    assert target["name"] == "Daniel"
    result = a.client.post(url, data={"entry_id": target["id"], "field":"Occupation", "value":"Watchmaker"},
                           follow_redirects=False)
    assert result.status_code == 303 and result.headers["location"].endswith("#message-1")
    assert "Watchmaker" in a.client.get(prefix + "/console/characters").text
    assert "Watchmaker" in a.main._book_notes(bid)
