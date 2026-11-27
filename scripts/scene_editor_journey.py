"""Exercise one continuous manuscript and inspect the requests sent to a local model."""

from pathlib import Path
import sys
import re

from playwright.sync_api import expect


def check_scenes(page, claw):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from provider_server import completion, provider_server

    page.goto(claw + "/console/chat")
    page.set_viewport_size({"width": 1360, "height": 900})
    page.locator("#browse-books").click()
    page.get_by_text("New book", exact=True).click()
    page.get_by_label("Book title").fill("Daniel scene test")
    page.get_by_role("button", name="Create book").click()
    book = page.url.split("/console/")[0]
    with provider_server(lambda *_: (200, completion("Scene received."))) as (
        url,
        requests,
    ):
        page.request.post(
            book + "/providers",
            form={
                "slug": "scene-test",
                "model": "scene-test",
                "base_url": url,
                "wire_api": "chat",
            },
        )
        page.request.post(
            book + "/api-keys",
            form={"provider": "scene-test", "api_key": "local-test-key"},
        )
        page.goto(book + "/console/manuscript/read")
        page.get_by_role("button", name="New chapter").click()
        prose = page.get_by_label("Chapter prose")
        text = "Morning at the mirror.\n\nAfternoon wish.\n\nEvening at home."
        prose.fill(text)
        page.get_by_label("Story day").fill("1")
        page.get_by_label("Time", exact=True).fill("09:00")

        def fact(value, phase):
            page.get_by_role("button", name="Add character state").click()
            page.get_by_label("Character", exact=True).fill("Daniel Turner")
            page.get_by_label("Trait", exact=True).fill("Height")
            page.get_by_label("Value", exact=True).fill(value)
            page.get_by_label("Applies").select_option(phase)

        def split_at(word, time):
            prose.evaluate(
                "(el, word) => {el.focus(); const n=el.value.indexOf(word); el.setSelectionRange(n,n)}",
                word,
            )
            page.get_by_role("button", name="Split at cursor").click()
            page.get_by_label("Time", exact=True).fill(time)

        fact("5′11″", "entry")
        split_at("Afternoon", "16:00")
        fact("4′2″", "change")
        split_at("Evening", "18:00")
        # Insert before both boundaries, then move the caret across them.
        prose.focus()
        prose.press("Control+Home")
        page.keyboard.insert_text("Daniel opens the jar. ")
        expect(page.locator("#scene-jump")).to_have_value("0")
        page.get_by_role("button", name="Save chapter").click()
        expect(page.locator("#scene-status")).to_have_text("Chapter saved")
        page.get_by_role("link", name="Read manuscript").click()
        page.get_by_role("link", name="Edit chapter & scenes").click()
        expect(prose).to_have_count(1)
        expect(prose).to_have_value(
            re.compile("Daniel opens the jar. Morning at the mirror")
        )
        page.get_by_text("What Claw sees", exact=True).click()
        page.get_by_text("Ask Claw about this scene", exact=True).click()
        for i in (1, 2, 0):
            page.locator("#scene-jump").select_option(str(i))
            page.get_by_label("Question or writing instruction").fill(
                "How tall is Daniel here?"
            )
            page.locator("#scene-provider").select_option("scene-test")
            page.get_by_role("button", name="Ask Claw", exact=True).click()
            expect(page.locator("#scene-answer")).to_have_text("Scene received.")
            messages = requests[-1]["body"]["messages"]
            packet = next(
                m["content"]
                for m in messages
                if m["role"] == "user" and m["content"].startswith("Chapter ")
            )
            assert packet == page.locator("#scene-context").text_content()
            wire = "\n".join(m["content"] for m in messages)
            if i == 0:
                assert "4′2″" not in wire
            assert ("5′11″" in packet) == (i != 2)
            assert ("4′2″" in packet) == (i != 0)
            if i == 0:
                assert (
                    "Afternoon wish." not in packet and "Evening at home." not in packet
                )
        assert len(requests) == 3
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth + 1"
        )
