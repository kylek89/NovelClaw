from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
from threading import Barrier

import pytest


@pytest.fixture
def idea_session(signed_in):
    if signed_in.name != "novelclaw":
        pytest.skip("NovelClaw conversation lifecycle")
    main = signed_in.main
    state = main.load_state("{}")
    state["refined_idea"] = "A lighthouse mystery."
    with signed_in.db.SessionLocal() as db:
        session = signed_in.models.IdeaCopilotSession(
            user_id=signed_in.user_id,
            provider="deepseek",
            original_idea="A lighthouse mystery.",
            status="active",
            conversation_json=main.dump_state(state),
        )
        db.add(session)
        db.commit()
        session_id = session.id
    return signed_in, session_id


def test_confirming_session_concurrently_creates_one_job(idea_session, monkeypatch):
    service, session_id = idea_session
    starts = []
    monkeypatch.setattr(service.main, "_start_generation_worker", starts.append)
    gate = Barrier(4)

    def confirm(_):
        with service.db.SessionLocal() as db:
            session = db.get(service.models.IdeaCopilotSession, session_id)
            gate.wait(timeout=5)
            return service.main._start_generation_from_idea_session(
                db, session, service.user_id
            ).id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(confirm, range(4)))
    assert len(set(ids)) == 1 and starts == [ids[0]]


def test_reply_cannot_overwrite_a_newer_brief(idea_session):
    service, session_id = idea_session
    with service.db.SessionLocal() as stale_db:
        stale = stale_db.get(service.models.IdeaCopilotSession, session_id)
        original = stale.conversation_json
        state = service.main.load_state(original)
        state.update(reply_pending=False, refined_idea="Stale model response")
        assert (
            service.client.post(
                f"/api/idea-copilot/{session_id}/brief",
                json={"brief": "Author's new direction"},
            ).status_code
            == 200
        )
        assert not service.main._save_reply_if_current(
            stale_db, stale, original, state, 100
        )
    with service.db.SessionLocal() as db:
        assert (
            db.get(service.models.IdeaCopilotSession, session_id).refined_idea
            == "Author's new direction"
        )


def test_reply_cannot_restart_a_canceled_session(idea_session, monkeypatch):
    service, session_id = idea_session
    main = service.main
    with service.db.SessionLocal() as db:
        session = db.get(service.models.IdeaCopilotSession, session_id)
        state = main.load_state(session.conversation_json)
        state["reply_pending"] = True
        session.conversation_json = main.dump_state(state)
        db.commit()
    monkeypatch.setattr(main, "_provider_api_key", lambda *args: "fake-test-key")
    monkeypatch.setattr(main, "_analyze_generation_intent", lambda **kw: {})
    monkeypatch.setattr(
        main,
        "_start_generation_worker",
        lambda *args: pytest.fail("canceled session restarted"),
    )

    class Agent:
        def __init__(self, *args):
            pass

        def generate_turn(self, **kwargs):
            service.client.post(
                f"/idea-copilot/{session_id}/cancel", follow_redirects=False
            )
            return {
                "role": "assistant",
                "analysis": "Start generation",
                "refined_idea": "Changed by model",
                "readiness": 100,
                "questions": [],
            }

    monkeypatch.setattr(main, "IdeaCopilotAgent", Agent)
    main._run_idea_copilot_reply_worker(session_id)
    with service.db.SessionLocal() as db:
        session = db.get(service.models.IdeaCopilotSession, session_id)
        assert session.status == "canceled" and session.final_job_id is None
        assert not main.load_state(session.conversation_json)["reply_pending"]


def test_help_with_previous_questions_stays_in_chat(idea_session, monkeypatch):
    import json
    from types import SimpleNamespace

    service, session_id = idea_session
    main = service.main
    agent = importlib.import_module(main.IdeaCopilotAgent.__module__)
    questions = ["What did the keeper lose?", "Why hide the log?", "Who finds the key?"]
    request = "Help me with the 3 questions."
    options = [["His compass", "His boat", "His job"], ["Debt", "His sister", "Shame"],
               ["The pilot", "The nurse", "The tailor"]]
    with service.db.SessionLocal() as db:
        session = db.get(service.models.IdeaCopilotSession, session_id)
        state = main.load_state(session.conversation_json)
        state.update(reply_pending=True, messages=[
            {"role": "assistant", "questions": questions},
            {"role": "user", "content": request},
            {"role": "assistant", "analysis": "A lighthouse mystery.", "questions": []},
            {"role": "user", "content": request},
        ])
        session.conversation_json = main.dump_state(state)
        db.commit()

    def respond(messages, **kwargs):
        prompt = json.loads(messages[-1]["content"])
        assert prompt["request"] == request and prompt["questions_in_order"] == questions
        return json.dumps({"question_options": options})

    monkeypatch.setattr(agent, "_build_client", lambda *a: SimpleNamespace(chat=respond))
    monkeypatch.setattr(main, "_model_generation_intent", lambda **kw: pytest.fail("Unnecessary intent call"))
    monkeypatch.setattr(main, "_provider_api_key", lambda *a: "fixture")
    monkeypatch.setattr(main, "_start_generation_from_idea_session", lambda *a: pytest.fail("Unexpected writing run"))
    main._run_idea_copilot_reply_worker(session_id)
    with service.db.SessionLocal() as db:
        session = db.get(service.models.IdeaCopilotSession, session_id)
        state = main.load_state(session.conversation_json)
        assert not state["reply_pending"] and not state["reply_error"]
        assert state["messages"][-1]["questions"] == questions
        assert state["messages"][-1]["question_options"] == options
        selected = agent.append_user_reply(state, "1B, 2A, 3C")
        assert selected["messages"][-1]["selected_options"] == [
            f"{questions[0]} — His boat", f"{questions[1]} — Debt", f"{questions[2]} — The tailor",
        ]
        brief = agent.build_generation_idea("A lighthouse mystery.", selected)
        assert "His boat" in brief and "His compass" not in brief
        assert not agent.valid_options([["Same", "same", "Third"]] * 3, questions)
        assert agent.requested_questions(state, "Give me three possible answers for each question.") == questions
        assert session.final_job_id is None


def test_two_reply_posts_launch_only_one_worker(idea_session, monkeypatch):
    service, session_id = idea_session
    starts = []
    gate = Barrier(2)
    monkeypatch.setattr(service.main, "_start_idea_copilot_reply_worker", starts.append)

    def key(*args):
        gate.wait(timeout=5)
        return "fake-key"

    monkeypatch.setattr(service.main, "_provider_api_key", key)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda n: service.client.post(
                    f"/idea-copilot/{session_id}/reply",
                    data={"reply": f"Revision {n}"},
                    headers={"X-Requested-With": "fetch"},
                ),
                range(2),
            )
        )
    assert sorted(response.status_code for response in responses) == [202, 409]
    assert starts == [session_id]
