"""Exercise the existing workspace using two books and a recording local model."""

import json
import re
import runpy
import sys
import time
from threading import Event
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect


def check_books(page, claw):
    root = Path(__file__).resolve().parents[1]
    smoke = runpy.run_path(str(root / "scripts" / "generation-smoke.py"))
    sys.path.insert(0, str(root / "tests"))
    from provider_server import completion, provider_server

    tags = ["CopperHeron", "SilverFinch"]
    worker_entered = Event()
    continue_worker = Event()

    def reply(path, body, number):
        tag = body["model"]
        if "Revise only this outline" in str(body):
            return 200, completion("The harbor closes before the second bell.")
        if "detect generation-start intent" in str(body):
            return 200, completion('{"should_start_generation": false}')
        if "collaborative ideation agent" in str(body):
            return 200, completion(
                json.dumps(
                    {
                        "analysis": tag + " conversation",
                        "refined_idea": tag + " mystery",
                        "questions": ["Who finds it?", "Where is it?", "What is missing?"],
                        "question_options": [["Keeper", "Pilot", "Doctor"], ["Dock", "Tower", "Boat"],
                                             ["Key", "Map", "Letter"]],
                        "readiness": 60,
                    }
                )
            )
        if tag == tags[0] and not continue_worker.is_set():
            worker_entered.set()
            if not continue_worker.wait(20):
                return 500, {"error": {"message": "Book switch did not complete"}}
        if "scene writing partner" in str(body):
            if "Reply with PASS" in str(body):
                return 200, completion("PASS")
            return 200, completion(smoke["STORY"].replace("Lin counted", tag + " counted"))
        code, result = smoke["fixture_reply"](path, body, number)
        return code, json.loads(
            json.dumps(result).replace("Lin counted", tag + " counted")
        )

    page.goto(claw + "/console/chat")
    page.set_viewport_size({"width": 1440, "height": 1000})
    prefixes = []
    sessions = []
    with provider_server(reply) as (url, requests):
        for i, tag in enumerate(tags):
            page.locator("#browse-books").click()
            page.get_by_text("New book", exact=True).click()
            page.get_by_label("Book title").fill(tag)
            page.get_by_role("button", name="Create book").click()
            expect(page.locator("#browse-books")).to_contain_text(tag)
            prefix = page.url.split("/console/")[0]
            prefixes.append(prefix)
            # Configure the test provider through the same forms used by Models.
            slug = "test-" + str(i)
            page.request.post(
                prefix + "/providers",
                form={"slug": slug, "model": tag, "base_url": url, "wire_api": "chat"},
            )
            page.request.post(
                prefix + "/api-keys",
                form={"provider": slug, "api_key": "local-test-key"},
            )
            nav = page.locator("#workspace-navigation")
            nav.get_by_role("link", name="13 Characters").click()
            page.get_by_text("Add character", exact=True).click()
            page.get_by_label("Name", exact=True).fill("Rowan")
            page.get_by_label("Profile", exact=True).fill(tag + " cartographer")
            page.get_by_role("button", name="Save character").click()
            expect(page.locator("main")).to_contain_text(tag + " cartographer")
            nav.get_by_role("link", name="07 Author Brief").click()
            page.get_by_label("Author Brief", exact=True).fill(tag + " keeps the brass compass")
            page.get_by_role("button", name="Save", exact=True).click()
            expect(page.get_by_text("History · 1")).to_be_visible()
            nav.locator('a[href*="/console/chat"]').first.click()
            page.locator("#provider-select").select_option(slug)
            page.locator("#new-session-form textarea").fill(
                tag + " story. Write one complete chapter."
            )
            page.get_by_role("button", name="Create conversation").click()
            expect(page.locator("main")).to_contain_text(tag + " conversation")
            cards = page.locator("[data-answer-card]")
            expect(cards).to_have_count(1)
            fields = cards.first.locator("fieldset")
            fields.nth(0).get_by_role("radio", name="B. Pilot").check()
            fields.nth(0).locator("select").select_option("premise")
            fields.nth(1).get_by_role("radio", name="Write my own").check()
            fields.nth(1).locator("textarea").fill(tag + " lost the harbor map")
            fields.nth(1).locator("select").select_option("author_brief")
            draft = page.locator("[data-idea-reply-form] textarea")
            draft.fill("An unsent thought")
            cards.first.get_by_role("button", name="Send answers").click()
            expect(cards).to_have_count(2, timeout=20000)
            expect(draft).to_have_value("An unsent thought")
            page.reload()
            expect(cards).to_have_count(2)
            expect(page.locator(".user-bubble").last).to_contain_text(tag + " lost the harbor map")
            assert "Who finds it? — Pilot" in page.request.get(prefix + "/console/memory/zone/premise").text()
            undo = page.locator('.user-bubble [data-doc-undo="premise"]').last
            undo.click()
            expect(page.locator(".user-bubble").last).to_contain_text("Undone")
            doc = page.context.new_page()
            doc.goto(prefix + "/console/memory/zone/premise")
            expect(doc.get_by_label("Story Premise", exact=True)).to_have_value("")
            doc.get_by_text("History · 2").click()
            doc.get_by_text("Revision 1 ·", exact=False).first.click()
            doc.get_by_role("button", name="Restore this version").click()
            expect(doc.get_by_text("History · 3")).to_be_visible()
            # An already-open editor cannot overwrite this newer revision.
            stale = page.request.post(prefix + "/console/memory/zone/premise/document",
                                      form={"version": "2", "content": "Obsolete rule"})
            assert stale.status == 409
            doc.get_by_text("Edit Story Premise").click()
            doc.get_by_label("Story Premise", exact=True).fill(tag + " follows the north beacon")
            doc.get_by_role("button", name="Save", exact=True).click()
            expect(doc.get_by_text("History · 4")).to_be_visible()
            doc.close()
            if i == 0:
                send = page.locator("[data-idea-reply-form]").get_by_role("button", name="Send", exact=True)
                for n in range(5):
                    draft.fill("What next?")
                    send.click()
                    expect(cards).to_have_count(n + 3, timeout=20000)
                draft.fill("1 - C\n2 - A\n3 - C")
                send.click()
                expect(cards).to_have_count(8, timeout=20000)
                page.reload()
                for choice in ("Who finds it? — Doctor", "Where is it? — Dock", "What is missing? — Letter"):
                    expect(page.locator(".user-bubble").last).to_contain_text(choice)
                for kind, rev in (("premise", 4), ("author_brief", 2)):
                    assert f"History · {rev}<" in page.request.get(prefix + "/console/memory/zone/" + kind).text()
            form = page.locator("[data-idea-reply-form]")
            number = cards.count()
            form.locator("[name=reply_destination]").select_option("chapter")
            form.get_by_label("Chapter number").fill("1")
            draft.fill(tag + " chapter-only instruction")
            form.get_by_role("button", name="Send", exact=True).click()
            expect(cards).to_have_count(number + 1, timeout=20000)
            page.reload()
            expect(page.get_by_text("Chapter directions (1)")).to_be_visible()
            if i == 0:
                form.locator("[name=reply_destination]").select_option("chapter")
                form.get_by_label("Chapter number").fill("2")
                draft.fill("Second-chapter-only marker")
                form.get_by_role("button", name="Send", exact=True).click()
                expect(cards).to_have_count(number + 2, timeout=20000)
                assert page.request.post(prefix + "/console/world", form={"name": "North Pier"}).ok
                group = cards.last.locator("fieldset")
                for field, destination in [(group.nth(0), "character"), (group.nth(1), "world")]:
                    field.get_by_role("radio").first.check()
                    field.locator("select").select_option(destination)
                cards.last.get_by_role("button", name="Send answers").click()
                expect(cards).to_have_count(number + 3, timeout=20000)
                for kind, name, field, detail in [("character", "Rowan", "Occupation", "Signal keeper"),
                                                   ("world", "North Pier", "Rule", "Closed after dusk")]:
                    page.get_by_role("link", name="Review " + kind + " detail").last.click()
                    expect(page.locator("#craft-review")).to_be_visible()
                    page.get_by_label("Entry", exact=True).select_option(label=name)
                    page.get_by_label("Field", exact=True).select_option(field)
                    page.get_by_label("Proposed value").fill(detail)
                    page.get_by_role("button", name="Save detail").click()
                    expect(page.locator(".user-bubble").last).to_contain_text("Review world detail")
                    assert detail in page.request.get(prefix + ("/console/characters" if kind == "character" else "/console/world")).text()
            sessions.append(parse_qs(urlsplit(page.url).query)["session_id"][0])
        # Run B through the authored-scene worker, A through the normal chapter writer.
        page.goto(prefixes[1] + "/console/manuscript/read")
        page.get_by_role("button", name="New chapter").click()
        page.get_by_label("Chapter prose").fill("She crossed the harbor.")
        page.get_by_role("button", name="Save chapter").click()
        expect(page.locator("#scene-status")).to_have_text("Chapter saved")
        # Keep two tabs open. Launch A, then work in B before A's worker completes.
        tab_a = page.context.new_page()
        tab_a.goto(prefixes[0] + "/console/chat")
        result = tab_a.request.post(
            prefixes[0] + "/idea-copilot/" + sessions[0] + "/confirm"
        )
        assert result.ok
        try:
            assert worker_entered.wait(15), (
                "First worker did not reach the recording provider"
            )
            check_delete = page.request.get(prefixes[0] + "/delete-book")
            token = re.search(
                r'name="confirmation" value="([^"]+)"', check_delete.text()
            )[1]
            busy = page.request.post(
                prefixes[0] + "/delete-book", form={"confirmation": token}
            )
            assert busy.status == 409 and "busy" in busy.text()
            page.goto(prefixes[0] + "/console/characters")
            page.locator("#browse-books").click()
            page.get_by_role("link", name=tags[1], exact=True).click()
            expect(page.locator("main")).to_contain_text(tags[1] + " cartographer")
            result = page.request.post(
                prefixes[1] + "/idea-copilot/" + sessions[1] + "/confirm"
            )
            assert result.ok
        finally:
            continue_worker.set()
        for i, tag in enumerate(tags):
            prefix = prefixes[i]
            deadline = time.monotonic() + 100
            while time.monotonic() < deadline:
                live = page.request.get(
                    prefix + "/api/idea-copilot/" + sessions[i] + "/live"
                ).json()
                if live["status"] == "succeeded":
                    break
                assert live["status"] not in ("failed", "canceled"), live.get(
                    "error_message"
                )
                time.sleep(0.25)
            else:
                raise AssertionError("Generation did not finish")
            page.goto(prefix + "/console/manuscript/read")
            payload = json.loads(page.locator("#workspace-bootstrap").text_content())[
                "data"
            ]
            assert tag + " counted" in str(payload["chapters"]), str(
                payload["chapters"]
            )[:180]
            assert tags[1 - i] not in json.dumps(payload)
            download = page.request.get(
                prefix
                + "/jobs/"
                + str(payload["download_job_id"])
                + "/download/chapter/1"
            )
            assert (
                download.ok
                and tag + " counted" in download.text()
                and tags[1 - i] not in download.text()
            )
            sessions_page = page.request.get(prefix + "/console/sessions").text()
            assert (
                tag + " mystery" in sessions_page
                and tags[1 - i] + " mystery" not in sessions_page
            )
        page.goto(prefixes[0] + "/console/characters")
        for i in (1, 0):
            page.locator("#browse-books").click()
            page.get_by_role("link", name=tags[i], exact=True).click()
            expect(page).to_have_url(prefixes[i] + "/console/characters")
            expect(page.locator("main")).to_contain_text(tags[i] + " cartographer")
            expect(page.locator("main")).not_to_contain_text(tags[1 - i])
            page.reload()
            expect(page.locator("#browse-books")).to_contain_text(tags[i])
        assert tab_a.url.startswith(prefixes[0] + "/")
        tab_a.close()
        for i, tag in enumerate(tags):
            calls = [r["body"] for r in requests if r["body"]["model"] == tag]
            assert len(calls) > 2
            messages = json.dumps([c["messages"] for c in calls])
            assert (
                tag + " cartographer" in messages
                and tag + " keeps the brass compass" in messages
            )
            assert tags[1 - i] not in messages
            prompts = ["\n".join(m["content"] for m in c["messages"] if isinstance(m.get("content"), str))
                       for c in calls]
            writer = "expert long-form fiction writer" if i == 0 else "scene writing partner"
            for signature in (writer, "full-book outline"):
                matching = [prompt for prompt in prompts if signature in prompt]
                assert matching, signature
                for prompt in matching:
                    assert "Second-chapter-only marker" not in prompt
                    assert prompt.count(tag + " follows the north beacon") == 1, signature
                    assert prompt.count(tag + " keeps the brass compass") == 1, signature
                    assert (tag + " chapter-only instruction" in prompt) == (signature == writer), signature
                    assert "Who finds it? — Pilot" not in prompt and "Tower" not in prompt, signature
        # A changed premise flags generated outlines. Proposing leaves the saved text intact.
        base = prefixes[0]
        page.goto(base + "/console/chat?new=1")
        expect(page.get_by_text("Chapter directions (1)")).to_be_visible()
        page.get_by_text("Chapter directions (1)").click()
        expect(page.locator("details").filter(has=page.get_by_text("Chapter directions (1)"))).to_contain_text("Second-chapter-only marker")
        assert page.request.post(base + "/console/memory/zone/premise/document",
                                 form={"version": 4, "content": "The harbor is now landlocked."}).ok
        page.goto(base + "/console/manuscript/outline")
        page.get_by_role("heading", name="Outlines to review").locator("..").get_by_role("link").first.click()
        review_url = page.url
        assert review_url.startswith(base + "/console/manuscript/outline/")
        assert page.request.get(review_url.replace(base, prefixes[1])).status == 404
        saved = page.get_by_label("Revised outline").input_value()
        page.get_by_text("Regenerate a proposal", exact=True).click()
        page.locator("#outline-provider").select_option("test-0")
        page.get_by_role("button", name="Generate proposal").click()
        expect(page.locator("#proposal-text")).to_have_text("The harbor closes before the second bell.", timeout=20000)
        expect(page.get_by_label("Revised outline")).to_have_value(saved)
        page.get_by_role("button", name="Use proposal in editor").click()
        expect(page.get_by_label("Revised outline")).to_have_value("The harbor closes before the second bell.")
        assert saved in page.request.get(review_url).text()
        page.get_by_role("button", name="Save revised outline").click()
        expect(page).to_have_url(base + "/console/manuscript/outline")
        page.goto(review_url)
        expect(page.get_by_label("Revised outline")).to_have_value("The harbor closes before the second bell.")
        expect(page.get_by_text("Reviewed against the current Book documents")).to_be_visible()
        page.goto(base + "/console/manuscript/outline")
        review_links = page.get_by_role("heading", name="Outlines to review").locator("..").get_by_role("link")
        if review_links.count():
            review_links.first.click()
            original = page.get_by_label("Revised outline").input_value()
            url_to_keep = page.url
            page.get_by_role("button", name="Keep saved outline").click()
            expect(page).to_have_url(base + "/console/manuscript/outline")
            page.goto(url_to_keep)
            expect(page.get_by_label("Revised outline")).to_have_value(original)
        # Exercise a second library page, series search and a canceled deletion.
        for idx in range(13):
            assert page.request.post(
                claw + "/books", form={"title": f"Extra book {idx:02}"}
            ).ok
        page.locator("#browse-books").click()
        expect(page.locator(".book-card")).to_have_count(12)
        page.get_by_role("link", name="Next", exact=True).click()
        expect(page.locator(".book-card")).to_have_count(3)
        page.get_by_label("Search by title or series").fill(tags[0])
        page.get_by_role("button", name="Search", exact=True).click()
        card = page.locator(".book-card")
        card.get_by_text("Set series").click()
        card.get_by_label("Series name").fill("Tide Cycle")
        card.get_by_role("button", name="Save series").click()
        page.get_by_label("Search by title or series").fill("Tide Cycle")
        page.get_by_role("button", name="Search", exact=True).click()
        expect(card).to_have_count(1)
        card.get_by_role("link", name="Delete", exact=True).click()
        expect(page.get_by_role("heading", level=1)).to_contain_text(tags[0])
        page.get_by_role("link", name="Cancel", exact=True).click()
        expect(page.get_by_role("link", name=tags[0], exact=True)).to_be_visible()
        page.locator(".book-card").filter(
            has=page.get_by_role("link", name=tags[0], exact=True)
        ).get_by_role("link", name="Delete", exact=True).click()
        page.get_by_role("button", name="Permanently delete book").click()
        expect(page.get_by_role("status")).to_contain_text(tags[0])
        assert page.request.get(prefixes[0] + "/console/chat").status == 404
        kept = page.request.get(prefixes[1] + "/console/manuscript/read").text()
        assert tags[1] + " counted" in kept and tags[0] + " counted" not in kept
        assert (
            tags[1] + " cartographer"
            in page.request.get(prefixes[1] + "/console/characters").text()
        )
        assert (
            tags[1] + " keeps the brass compass"
            in page.request.get(
                prefixes[1] + "/console/memory/zone/author_brief"
            ).text()
        )
        assert (
            tags[1] + " conversation"
            in page.request.get(prefixes[1] + "/console/chat").text()
        )
        print(
            "Two-book browser, concurrent workers, chapter downloads and recorded prompts passed.",
            flush=True,
        )
