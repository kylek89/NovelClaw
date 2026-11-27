"""Check the same model controls across settings and the chat composer."""

from pathlib import Path
import sys
from playwright.sync_api import expect


def check_models(page, claw):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from provider_server import provider_server, completion

    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(claw + "/console/chat")
    page.locator("#browse-books").click()
    page.get_by_text("New book", exact=True).click()
    page.get_by_label("Book title").fill("Model picker fixture")
    page.get_by_role("button", name="Create book").click()
    book = page.url.split("/console/")[0]
    rows = [
        {"id": "fixture-a", "context_length": 4096},
        {"id": "fixture-b", "supported_parameters": ["tools"]},
    ]
    with provider_server(
        lambda *_: (
            200,
            completion(
                '{"analysis":"A quiet opening.","questions":["When?"],"readiness":40}'
            ),
        ),
        models=rows,
    ) as (url, calls):
        page.goto(book + "/console/models")
        form = page.locator("form").filter(has=page.locator('input[name="slug"]'))
        form.locator('[name="kind"]').select_option("local")
        for name, value in {
            "slug": "test-local",
            "label": "Local fixture",
            "base_url": url,
            "model": "fixture-a",
        }.items():
            form.locator(f'[name="{name}"]').fill(value)
        form.get_by_role("button").click()
        picker = page.locator("[data-model-picker]")
        picker.locator("select").select_option("test-local")
        picker.get_by_role("button", name="Test connection").click()
        expect(picker.locator("[data-model-info]")).to_contain_text(
            "Connected · 2 models"
        )
        picker.get_by_label("Model", exact=True).fill("fixture-b")
        picker.get_by_role("button", name="Favorite model").click()
        expect(picker.locator("[data-favorite]")).to_have_attribute(
            "aria-pressed", "true"
        )
        page.goto(book + "/console/chat?new=1")
        picker = page.locator("[data-model-picker]")
        picker.locator("select").select_option("test-local")
        expect(picker.locator("datalist option").first).to_have_attribute(
            "value", "fixture-b"
        )
        picker.get_by_label("Model", exact=True).fill("fixture-b")
        page.locator("#new-session-form textarea").fill("Plan a lighthouse story.")
        page.get_by_role("button", name="Create conversation").click()
        expect(page.get_by_text("Model · fixture-b", exact=True)).to_be_visible()
        assert calls[-1]["body"]["model"] == "fixture-b"
