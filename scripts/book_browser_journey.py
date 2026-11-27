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

    book.get_by_role("link", name="Book library", exact=True).click()
    restored_id = new_url.split("/")[-2]
    restored_row = book.locator(f'form[data-book="{restored_id}"]')
    page.once("dialog", lambda dialog: dialog.dismiss())
    restored_row.get_by_role("button").click()
    expect(restored_row).to_be_visible()
    page.once("dialog", lambda dialog: dialog.accept())
    restored_row.get_by_role("button").click()
    expect(restored_row).to_have_count(0)
    assert page.request.get(urljoin(page.url, new_url)).status == 404
    assert page.request.get(urljoin(page.url, backup_url)).json() == original

    page.set_viewport_size({"width": 1440, "height": 1000})
    sidebar = page.locator("#workspace-navigation")
    sidebar.get_by_role("link", name="Characters", exact=True).click()
    expect(book.get_by_role("heading", name="Choose a Book for Characters")).to_be_visible()
    book.get_by_role("link", name="Ferry rehearsal").click()
    expect(book.get_by_label("Character name", exact=True)).to_have_value("Keeper")
    book.get_by_label("Baseline profile").fill("Keeps the pier lantern lit.")
    book.get_by_role("button", name="Save character").click()
    expect(book.locator("[data-edit=character] textarea[name=text]")).to_have_value("Keeps the pier lantern lit.")
    sidebar.get_by_role("link", name="Storyboard", exact=True).click()
    book.get_by_role("link", name="Cast", exact=True).click()
    expect(book.get_by_label("Baseline profile")).to_have_value("Keeps the pier lantern lit.")
    first_book_url = page.url
    sidebar.get_by_role("link", name="Book library", exact=True).click()
    book.get_by_label("Book title").fill("Second ferry")
    book.get_by_role("button", name="Create Book").click()
    expect(book.get_by_role("heading", name="Second ferry")).to_be_visible()
    sidebar.get_by_role("link", name="Characters", exact=True).click()
    expect(book.get_by_role("heading", name="Cast · 0")).to_be_visible()
    book.get_by_label("New character name").fill("Pilot")
    book.get_by_role("button", name="Add character").click()
    expect(book.get_by_label("Character name", exact=True)).to_have_value("Pilot")
    first = page.context.new_page()
    first.goto(first_book_url)
    expect(first.get_by_label("Character name", exact=True)).to_have_value("Keeper")
    expect(first.get_by_label("Baseline profile")).to_have_value("Keeps the pier lantern lit.")
    first.close()
