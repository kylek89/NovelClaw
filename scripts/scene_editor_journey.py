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
        page.goto(book + "/console/style")
        page.get_by_label("Default intimacy level").select_option("sensual")
        page.get_by_label("Vocabulary, tone and boundaries").fill("Keep the language spare.")
        page.get_by_role("button", name="Save prose settings").click()
        expect(page.get_by_role("status").filter(has_text="Book prose settings saved.")).to_be_visible()
        page.goto(book + "/console/characters")
        page.get_by_text("Add character", exact=True).click()
        page.get_by_label("Name", exact=True).fill("Daniel Turner")
        page.get_by_label("Profile", exact=True).fill("A keeper.")
        page.get_by_role("button", name="Save character", exact=True).click()
        page.get_by_text("Intimate character note", exact=True).click()
        page.get_by_label("Private note for Daniel Turner").fill("Private scene marker.")
        page.get_by_role("button", name="Save intimate note").click()
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

        page.get_by_label("Intimacy", exact=True).select_option("closed")
        page.get_by_label("POV", exact=True).fill("Daniel Turner")
        expect(page.get_by_label("Include intimate notes for this cast")).to_be_disabled()
        fact("5′11″", "entry")
        split_at("Afternoon", "16:00")
        fact("4′2″", "change")
        split_at("Evening", "18:00")
        page.get_by_label("Scene cast").fill("Daniel Turner")
        page.get_by_label("Include intimate notes for this cast").check()
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
            assert "Keep the language spare." in packet
            assert ("Private scene marker." in packet) == (i == 2)
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
        for action in ("Draft scene", "Review scene", "Revise scene"):
            page.get_by_label("Action", exact=True).select_option(label=action)
            page.get_by_label("Question or writing instruction").fill("")
            with page.expect_response(lambda res: res.url.endswith("/ask") and res.request.method == "POST"):
                page.get_by_role("button", name=action, exact=True).click()
            expect(page.locator("#scene-answer")).to_have_text("Scene received.")
            assert "5′11″" in str(requests[-1]) and "4′2″" not in str(requests[-1])
        page.get_by_role("button", name="Use as scene prose").click()
        page.get_by_role("button", name="Save chapter").click()
        expect(page.locator("#scene-status")).to_have_text("Chapter saved")
        page.reload()
        expect(prose).to_have_value(re.compile(r"^Scene received\.[\s\S]*Afternoon wish\.[\s\S]*Evening at home"))
        assert len(requests) == 6
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth + 1"
        )
