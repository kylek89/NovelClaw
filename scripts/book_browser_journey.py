"""Exercise the author pages inside the browser smoke test's disposable workspace."""

import json
from urllib.parse import urljoin

from playwright.sync_api import expect


def check_book_workspace(page, portal):
    page.goto(portal + "/books")
    book = page.locator("#author-book")
    book.get_by_label("Book title").fill("Ferry rehearsal")
    book.get_by_label("Brain dump").fill("A keeper and a ferry.")
    book.get_by_role("button", name="Create Book").click()
    expect(book.get_by_role("heading", name="Ferry rehearsal")).to_be_visible()
    book.get_by_role("button", name="Add chapter").click()
    book.get_by_label("Chapter outline").fill("The keeper waits.")
    book.get_by_role("button", name="Save outline").click()
    expect(book.get_by_label("Chapter outline")).to_have_value("The keeper waits.")
    book.get_by_role("link", name="Storyboard", exact=True).click()
    book.get_by_role("link", name="Chapter 1 prose").click()
    prose = book.get_by_label("Scene prose")
    prose.fill("Morning. Night.")
    prose.evaluate("el => el.setSelectionRange(8, 8)")
    book.get_by_role("button", name="Split at cursor").click()
    expect(prose).to_have_value(" Night.")
    book.get_by_label("Story day / time").fill("Day 1, dusk")
    book.get_by_label("POV", exact=True).fill("Keeper")
    book.get_by_role("button", name="Save scene").click()
    expect(book.get_by_label("Story day / time")).to_have_value("Day 1, dusk")
    book.get_by_role("link", name="Cast", exact=True).click()
    book.get_by_label("New character name").fill("Keeper")
    book.get_by_role("button", name="Add character").click()
    book.get_by_text("Record a change", exact=True).click()
    book.get_by_label("Source scene").select_option(index=1)
    book.get_by_label("New state").fill("At the pier")
    book.get_by_label("Exact source excerpt").fill("Night.")
    book.get_by_role("button", name="Approve change").click()
    expect(book.get_by_text("At the pier", exact=False)).to_be_visible()
    book.get_by_label("As of scene").select_option(index=1)
    book.get_by_role("button", name="View state").click()
    expect(book.get_by_text("No approved changes in this range.")).to_be_visible()
    book.get_by_role("button", name="Develop with Claw").click()
    expect(book.get_by_role("heading", name="Develop Keeper")).to_be_visible()
    book.get_by_role("link", name="Storyboard", exact=True).click()
    book.get_by_role("link", name="Chapter 1 outline").click()
    expect(book.get_by_label("Chapter outline")).to_have_value("The keeper waits.")
    backup_url = book.get_by_role("link", name="Back up Book").get_attribute("href")
    original = page.request.get(urljoin(page.url, backup_url)).json()
    book.get_by_role("link", name="Book library", exact=True).click()
    book.get_by_label("Book backup").set_input_files(
        {
            "name": "ferry.json",
            "mimeType": "application/json",
            "buffer": json.dumps(original).encode(),
        }
    )
    book.get_by_role("button", name="Restore Book").click()
    expect(book.get_by_role("heading", name="Ferry rehearsal")).to_be_visible()
    new_url = book.get_by_role("link", name="Back up Book").get_attribute("href")
    assert (
        new_url != backup_url
        and page.request.get(urljoin(page.url, new_url)).json() == original
    )
