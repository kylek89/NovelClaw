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
    with pytest.raises(ValueError):
        main.resolve_question_answers(state, {"questions": list(reversed(qs)), "options": opts, "picks": [2, None, None]})
