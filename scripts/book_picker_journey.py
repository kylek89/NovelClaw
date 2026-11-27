"""Exercise the existing workspace using two books and a recording local model."""

import json
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
        if "collaborative ideation agent" in str(body):
            return 200, completion(
                json.dumps(
                    {
                        "analysis": tag + " conversation",
                        "refined_idea": tag + " mystery",
                        "questions": ["Who finds it?"],
                        "readiness": 60,
                    }
                )
            )
        if tag == tags[0] and not continue_worker.is_set():
            worker_entered.set()
            if not continue_worker.wait(20):
                return 500, {"error": {"message": "Book switch did not complete"}}
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
            picker = page.locator(".book-picker")
            if not picker.locator("details").evaluate("el => el.open"):
                picker.locator("summary").click()
            picker.get_by_label("Book title").fill(tag)
            picker.get_by_role("button", name="Create book").click()
            expect(page.locator("#active-book option:checked")).to_have_text(tag)
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
            page.locator("#memory-add-content").fill(tag + " keeps the brass compass")
            page.locator("#memory-add-form button").click()
            expect(page.locator("#memory-add-content")).to_have_value("")
            nav.locator('a[href*="/console/chat"]').first.click()
            page.locator("#provider-select").select_option(slug)
            page.locator("#new-session-form textarea").fill(
                tag + " story. Write one complete chapter."
            )
            page.get_by_role("button", name="Create conversation").click()
            expect(page.locator("main")).to_contain_text(tag + " conversation")
            sessions.append(parse_qs(urlsplit(page.url).query)["session_id"][0])
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
            page.goto(prefixes[0] + "/console/characters")
            page.get_by_label("Book", exact=True).select_option(label=tags[1])
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
            page.get_by_label("Book", exact=True).select_option(label=tags[i])
            expect(page).to_have_url(prefixes[i] + "/console/characters")
            expect(page.locator("main")).to_contain_text(tags[i] + " cartographer")
            expect(page.locator("main")).not_to_contain_text(tags[1 - i])
            page.reload()
            expect(page.locator("#active-book option:checked")).to_have_text(tags[i])
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
        print(
            "Two-book browser, concurrent workers, chapter downloads and recorded prompts passed.",
            flush=True,
        )
